import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
import torch.distributed as dist
from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise
from bg_augmentation import apply_aug_split_batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EmbeddingQueue:
    def __init__(self, size: int, dim: int, device: torch.device):
        self.size = int(size)
        self.dim = int(dim)
        self.device = device
        self.queue = torch.zeros(self.size, self.dim, device=device)
        self.labels = torch.zeros(self.size, device=device, dtype=torch.long)
        self.ptr = 0
        self.is_full = False

    def __len__(self) -> int:
        return self.size if self.is_full else self.ptr

    def get(self):
        if len(self) == 0:
            return None, None
        if self.is_full:
            return self.queue, self.labels
        return self.queue[: self.ptr], self.labels[: self.ptr]

    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if embeddings is None or embeddings.numel() == 0:
            return
        emb = embeddings.detach()
        lab = labels.detach().long()
        n = emb.size(0)
        if n >= self.size:
            self.queue.copy_(emb[-self.size:])
            self.labels.copy_(lab[-self.size:])
            self.ptr = 0
            self.is_full = True
            return
        end = self.ptr + n
        if end <= self.size:
            self.queue[self.ptr:end] = emb
            self.labels[self.ptr:end] = lab
        else:
            first = self.size - self.ptr
            self.queue[self.ptr:] = emb[:first]
            self.labels[self.ptr:] = lab[:first]
            remain = n - first
            self.queue[:remain] = emb[first:]
            self.labels[:remain] = lab[first:]
        self.ptr = (self.ptr + n) % self.size
        if not self.is_full and self.ptr == 0:
            self.is_full = True


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, seed: int = 0, rank: int = 0, world_size: int = 1):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.data = dataset.data
        self.real = [i for i, it in enumerate(self.data) if it[1] == 1]
        self.fake = [i for i, it in enumerate(self.data) if it[1] == 0]
        self.per_class = batch_size // 2

        self.num_batches_total = min(len(self.real)//self.per_class, len(self.fake)//self.per_class)

        # IMPORTANT: make it divisible so every rank has SAME number of steps
        self.num_batches_total = (self.num_batches_total // world_size) * world_size
        self.num_batches_per_rank = self.num_batches_total // world_size

        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        return self.num_batches_per_rank

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(self.real)
        rng.shuffle(self.fake)

        # truncate to exactly the usable amount
        r = self.real[: self.num_batches_total * self.per_class]
        f = self.fake[: self.num_batches_total * self.per_class]

        # each rank takes its own interleaved batches
        for local_b in range(self.num_batches_per_rank):
            b = local_b * self.world_size + self.rank  # global batch index
            idx = r[b*self.per_class:(b+1)*self.per_class] + f[b*self.per_class:(b+1)*self.per_class]
            rng.shuffle(idx)
            yield idx

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def apply_rawboost_batch(x: torch.Tensor, attn_mask: torch.Tensor, cfg) -> torch.Tensor:
    if not cfg.use_rawboost:
        return x
    device = x.device

    a = x.detach().cpu().numpy()
    for i in range(a.shape[0]):
        if random.random() < cfg.rawboost_prob:
            xi = a[i]
            original_shape = xi.shape
            xi_1d = xi.ravel()
            target_len = xi_1d.shape[0]
            y = LnL_convolutive_noise(
                xi_1d, N_f=5, nBands=5,
                minF=20.0,  maxF=8000.0,
                minBW=100.0, maxBW=1000.0,
                minCoeff=10, maxCoeff=100,
                minG=0.0, maxG=0.0,
                minBiasLinNonLin=5.0, maxBiasLinNonLin=20.0,
                fs=cfg.target_sample_rate,
            )
            if random.random() < 0.5:
                y = SSI_additive_noise(
                    y, SNRmin=10.0, SNRmax=40.0, nBands=5,
                    minF=20.0, maxF=8000.0, minBW=100.0, maxBW=1000.0,
                    minCoeff=10, maxCoeff=100, minG=0.0, maxG=0.0,
                    fs=cfg.target_sample_rate,
                )
            if random.random() < 0.5:
                y = ISD_additive_noise(y, P=10.0, g_sd=2.0)
            if y.shape[0] > target_len:
                y = y[:target_len]
            elif y.shape[0] < target_len:
                y = np.pad(y, (0, target_len - y.shape[0]))
            a[i] = y.reshape(original_shape)

    return torch.from_numpy(a).to(device=device, dtype=x.dtype)

def alpha_for_epoch(epoch: int, cfg) -> float:
    if epoch <= cfg.warmup_epochs:
        return 0.0
    t = min(1.0, (epoch - cfg.warmup_epochs) / max(1, cfg.alpha_ramp_epochs))
    return t * cfg.alpha_end


def _concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)


def _reduce_avg(total: float, steps: int, device: torch.device) -> float:
    avg = total / max(1, steps)
    if dist.is_initialized():
        total_t = torch.tensor(total, device=device)
        steps_t = torch.tensor(steps, device=device)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps_t, op=dist.ReduceOp.SUM)
        avg = (total_t / steps_t.clamp_min(1)).item()
    return avg


def train_one_epoch(encoder, head, loss_fn, loader, optimizer, device, epoch, cfg, queue=None):
    if cfg.finetune_encoder:
        encoder.train()
    else:
        encoder.eval()
    head.train()
    total, steps = 0.0, 0
    alpha = alpha_for_epoch(epoch, cfg)
    LOG_EVERY = 500
    rank = dist.get_rank() if dist.is_initialized() else 0
    use_amp = bool(getattr(cfg, "use_amp", False)) and torch.cuda.is_available()
    if use_amp and not hasattr(cfg, "_amp_scaler"):
        cfg._amp_scaler = torch.cuda.amp.GradScaler()
    accum_steps = int(getattr(cfg, "accum_steps", 1))
    if accum_steps < 1:
        accum_steps = 1
    optimizer.zero_grad(set_to_none=True)
    for step_idx, (waveforms, bin_labels, attack_ids, *_ ) in enumerate(loader, start=1):
        waveforms = waveforms.to(device)
        raw_labels = attack_ids if getattr(cfg, 'label_type', 'binary') == 'attack_type' else bin_labels
        labels = raw_labels.to(device).long()

        if cfg.use_bg_aug:
            waveforms = apply_aug_split_batch(waveforms, cfg)
        elif cfg.use_rawboost:
            waveforms = apply_rawboost_batch(waveforms, labels, cfg)

        # windowed loader returns (B, N_chunks, T); take first chunk and compute proper mask
        if waveforms.ndim == 3:
            waveforms = waveforms[:, 0, :]
        attn = (waveforms != 0.0).long()

        queue_embs, queue_labels = (None, None)
        if queue is not None:
            queue_embs, queue_labels = queue.get()

        if use_amp:
            with torch.cuda.amp.autocast():
                if cfg.finetune_encoder:
                    hs = encoder(waveforms, attention_mask=attn)
                else:
                    with torch.no_grad():
                        hs = encoder(waveforms, attention_mask=attn)
                seq = head(hs)
                z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
                if queue_embs is None or queue_labels is None:
                    loss = loss_fn(z, labels, topk_neg=cfg.topk_neg, alpha=alpha)
                else:
                    loss = loss_fn(z, labels, topk_neg=cfg.topk_neg, alpha=alpha,
                                   queue_embs=queue_embs, queue_labels=queue_labels)
        else:
            if cfg.finetune_encoder:
                hs = encoder(waveforms, attention_mask=attn)
            else:
                with torch.no_grad():
                    hs = encoder(waveforms, attention_mask=attn)
            seq = head(hs)
            z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
            if queue_embs is None or queue_labels is None:
                loss = loss_fn(z, labels, topk_neg=cfg.topk_neg, alpha=alpha)
            else:
                loss = loss_fn(z, labels, topk_neg=cfg.topk_neg, alpha=alpha,
                               queue_embs=queue_embs, queue_labels=queue_labels)

        with torch.no_grad():
            B = labels.numel()
            sim = z @ z.t()
            eye = torch.eye(B, device=z.device, dtype=torch.bool)

            pos = (labels.view(-1,1) == labels.view(1,-1)) & (~eye)
            neg = (labels.view(-1,1) != labels.view(1,-1))

            pos_mean = sim[pos].mean().item() if pos.any() else float("nan")
            neg_mean = sim[neg].mean().item() if neg.any() else float("nan")

            if steps % LOG_EVERY == 0 and rank == 0:
                print(f"[rank{rank}] pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} "
                    f"n_pos={int((labels==1).sum())} n_neg={int((labels==0).sum())}")

        if steps % LOG_EVERY == 0:
            with torch.no_grad():
                sim = z @ z.t()
                print(f"[rank{rank}] sim_mean={sim.mean().item():.4f} sim_std={sim.std().item():.4f} Loss: {loss}")

        loss = loss / accum_steps
        if use_amp:
            cfg._amp_scaler.scale(loss).backward()
        else:
            loss.backward()

        if queue is not None:
            z_store = _concat_all_gather(z.detach())
            y_store = _concat_all_gather(labels.detach())
            queue.enqueue(z_store, y_store)

        if step_idx % accum_steps == 0:
            if use_amp:
                cfg._amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                cfg._amp_scaler.step(optimizer)
                cfg._amp_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total += loss.item() * accum_steps
        steps += 1

    return _reduce_avg(total, steps, device), alpha


@torch.no_grad()
def evaluate(encoder, head, loss_fn, loader, device, cfg):
    encoder.eval()
    head.eval()
    total, steps = 0.0, 0
    use_amp = bool(getattr(cfg, "use_amp", False)) and torch.cuda.is_available()
    for waveforms, bin_labels, attack_ids, *_ in loader:
        waveforms = waveforms.to(device)
        raw_labels = attack_ids if getattr(cfg, 'label_type', 'binary') == 'attack_type' else bin_labels
        labels = raw_labels.to(device).long()

        if waveforms.ndim == 3:
            waveforms = waveforms[:, 0, :]
        attn = (waveforms != 0.0).long()

        if use_amp:
            with torch.cuda.amp.autocast():
                hs = encoder(waveforms, attention_mask=attn)
                seq = head(hs)
                z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        else:
            hs = encoder(waveforms, attention_mask=attn)
            seq = head(hs)
            z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        loss = loss_fn(z, labels, topk_neg=cfg.topk_neg, alpha=0.0)

        total += loss.item()
        steps += 1
    return total / max(1, steps)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    else:
        return False, 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0
