"""
train_aug_ntxent.py — Stage 1 NT-Xent training with multi-view augmentation.

Differences from wav2vec_contr_loss/train_stage1.py:
  - Loss: NTXentMultiPositiveLoss (self-supervised, no labels)
  - Per-batch augmentation: AudioAugmentor generates n_views views per utterance
  - Encoder fine-tuned by default (--freeze_encoder 0); use --freeze_encoder 1 to freeze
  - Dev loss computed on bona fide-only samples

Usage (single GPU):
    python train_aug_ntxent.py [args]

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=2 train_aug_ntxent.py [args]
"""

import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from asvspoof_windowed_loader import ASVspoof2019WindowedDataset
from collate import pad_collate_fn_speaker_source_multiclass
from stage1_utils import BalancedBatchSampler, set_seed, setup_distributed
from torch.utils.data.distributed import DistributedSampler

from aug_config import build_config
from augmentations import AudioAugmentor
from losses import NTXentMultiPositiveLoss


# ---------------------------------------------------------------------------
# DDP gather — preserves gradient through local rank's slice
# ---------------------------------------------------------------------------

def gather_embeddings(z_flat: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    Gather z_flat (B_local*n_views, D) from all ranks.
    The local rank's slice retains gradients; all other slices are detached.
    Returns (B_global*n_views, D).
    """
    gathered = [torch.zeros_like(z_flat) for _ in range(world_size)]
    dist.all_gather(gathered, z_flat.detach())
    gathered[rank] = z_flat  # restore grad-carrying copy for local slice
    return torch.cat(gathered, dim=0)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def _forward_pass(encoder, head, views_flat, attn_mask, freeze_encoder: bool):
    if freeze_encoder:
        with torch.no_grad():
            hs = encoder(views_flat, attention_mask=attn_mask)
    else:
        hs = encoder(views_flat, attention_mask=attn_mask)
    seq = head(hs)                                    # (B*n_views, 256, T_feat)
    z_flat = F.normalize(seq.mean(dim=-1), p=2, dim=1)  # (B*n_views, 256)
    return z_flat


def train_one_epoch_ntxent(
    encoder, head, loss_fn, augmentor, loader,
    optimizer, device, epoch, cfg,
    is_distributed, rank, world_size,
):
    encoder.train() if not cfg.freeze_encoder else encoder.eval()
    head.train()

    total_loss, steps = 0.0, 0

    for batch in loader:
        waveforms = batch[0]   # (B, T_max)
        views_cpu = augmentor.get_views_batch(waveforms.cpu(), n=cfg.n_views)  # (B, n_views, T)
        B, n_views, T = views_cpu.shape
        views_flat = views_cpu.view(B * n_views, T).to(device)
        attn_mask  = (views_flat != 0.0).long()

        z_flat = _forward_pass(encoder, head, views_flat, attn_mask, bool(cfg.freeze_encoder))

        if is_distributed:
            z_flat = gather_embeddings(z_flat, rank, world_size)
            B_global = B * world_size
        else:
            B_global = B

        z = z_flat.view(B_global, n_views, -1)   # (N, n_views, D)
        loss = loss_fn(z) / cfg.accum_steps

        loss.backward()

        if (steps + 1) % cfg.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + (list(encoder.parameters()) if not cfg.freeze_encoder else []),
                max_norm=5.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * cfg.accum_steps
        steps += 1

    # Final step if steps not divisible by accum_steps
    remaining = steps % cfg.accum_steps
    if remaining != 0:
        torch.nn.utils.clip_grad_norm_(
            list(head.parameters()) + (list(encoder.parameters()) if not cfg.freeze_encoder else []),
            max_norm=5.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_ntxent(
    encoder, head, loss_fn, augmentor, dev_dataset,
    device, cfg, is_distributed, rank, world_size,
):
    """
    Evaluate NT-Xent loss on bona fide-only dev samples.
    Spoof utterances are excluded — they are not semantically equivalent views
    and would corrupt the NT-Xent denominator as a model-selection signal.
    """
    encoder.eval()
    head.eval()

    bonafide_indices = [i for i, row in enumerate(dev_dataset.data) if row[1] == 1]
    if not bonafide_indices:
        return float("nan")

    bonafide_ds = Subset(dev_dataset, bonafide_indices)
    dev_loader = DataLoader(
        bonafide_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
        pin_memory=True,
    )

    total_loss, steps = 0.0, 0
    for batch in dev_loader:
        views_cpu = augmentor.get_views_batch(batch[0].cpu(), n=cfg.n_views)
        B, n_views, T = views_cpu.shape
        views_flat = views_cpu.view(B * n_views, T).to(device)
        attn_mask = (views_flat != 0.0).long()

        hs = encoder(views_flat, attention_mask=attn_mask)
        seq = head(hs)
        z_flat = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        z = z_flat.view(B, n_views, -1)
        loss = loss_fn(z)
        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_ckpt(path, epoch, encoder, head, train_loss, dev_loss, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "compression_state_dict": (
            head.module.state_dict() if isinstance(head, DDP) else head.state_dict()
        ),
        "config": vars(cfg),
    }
    if not cfg.freeze_encoder:
        ckpt["encoder_state_dict"] = (
            encoder.module.state_dict() if isinstance(encoder, DDP) else encoder.state_dict()
        )
    torch.save(ckpt, path)


def _model_tag(model_name: str) -> str:
    return model_name.replace("/", "__")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = build_config()

    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed + rank)

    bonafide_only = bool(cfg.train_bonafide_only)

    if rank == 0:
        print(f"[Config] model={cfg.model_name}  n_views={cfg.n_views}  "
              f"temperature={cfg.temperature}  aug_mode={cfg.aug_mode}")
        print(f"[Config] freeze_encoder={cfg.freeze_encoder}  "
              f"batch_size={cfg.batch_size}  epochs={cfg.epochs}")
        print(f"[Config] noise_dir={cfg.noise_dir}  rir_dir={cfg.rir_dir}")
        print(f"[Config] train_bonafide_only={bonafide_only}")

    # ------------------------------------------------------------------
    # Augmentor
    # ------------------------------------------------------------------
    augmentor = AudioAugmentor(
        noise_dir=cfg.noise_dir or None,
        rir_dir=cfg.rir_dir or None,
        n_views=cfg.n_views,
        mode=cfg.aug_mode,
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    num_samples = cfg.num_samples

    train_subset = "bonafide" if bonafide_only else "all"
    train_ds = ASVspoof2019WindowedDataset(
        protocol_file=cfg.train_protocol,
        root_dir=cfg.train_root,
        subset=train_subset,
        num_samples=num_samples,
        sample_seed=cfg.seed,
        target_sample_rate=16000,
        window_seconds=cfg.max_duration_seconds,
    )
    dev_ds = ASVspoof2019WindowedDataset(
        protocol_file=cfg.dev_protocol,
        root_dir=cfg.dev_root,
        subset="all",
        target_sample_rate=16000,
        window_seconds=cfg.max_duration_seconds,
    )

    if rank == 0:
        print(f"[Data] train={len(train_ds)} ({train_subset})  dev={len(dev_ds)}")

    # ------------------------------------------------------------------
    # DataLoader  (workers load audio only; augmentation happens in main process)
    # ------------------------------------------------------------------
    train_sampler = None  # set below only when a sampler is needed
    if bonafide_only:
        if is_distributed:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                sampler=train_sampler,
                num_workers=cfg.num_workers,
                collate_fn=pad_collate_fn_speaker_source_multiclass,
                pin_memory=True,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                collate_fn=pad_collate_fn_speaker_source_multiclass,
                pin_memory=True,
                drop_last=True,
            )
    else:
        train_sampler = BalancedBatchSampler(
            train_ds, cfg.batch_size, seed=cfg.seed, rank=rank, world_size=world_size
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=cfg.num_workers,
            collate_fn=pad_collate_fn_speaker_source_multiclass,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    encoder = Wav2Vec2Encoder(
        model_name=cfg.model_name,
        freeze_encoder=bool(cfg.freeze_encoder),
    ).to(device)

    head = CompressionModule(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        dropout_rate=cfg.dropout,
        use_bottleneck=bool(cfg.use_bottleneck),
    ).to(device)

    if is_distributed:
        encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
        head    = DDP(head,    device_ids=[local_rank], find_unused_parameters=False)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    loss_fn = NTXentMultiPositiveLoss(temperature=cfg.temperature).to(device)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    param_groups = [{"params": head.parameters(), "lr": cfg.head_lr}]
    if not cfg.freeze_encoder:
        param_groups.append({"params": encoder.parameters(), "lr": cfg.enc_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    best_dev_loss = float("inf")
    patience_counter = 0

    model_tag = _model_tag(cfg.model_name)
    save_root = os.path.join(cfg.save_dir, model_tag)
    best_path  = os.path.join(save_root, f"{model_tag}_ntxent_best.pt")
    last_path  = os.path.join(save_root, f"{model_tag}_ntxent_last.pt")

    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        _enc = encoder.module if isinstance(encoder, DDP) else encoder
        _hd  = head.module    if isinstance(head, DDP)    else head
        if "encoder_state_dict" in ckpt:
            _enc.load_state_dict(ckpt["encoder_state_dict"])
        _hd.load_state_dict(ckpt["compression_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dev_loss = ckpt.get("dev_loss", float("inf"))
        if rank == 0:
            print(f"[Resume] epoch={start_epoch - 1}  best_dev_loss={best_dev_loss:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss = train_one_epoch_ntxent(
            encoder, head, loss_fn, augmentor, train_loader,
            optimizer, device, epoch, cfg,
            is_distributed, rank, world_size,
        )
        dev_loss = evaluate_ntxent(
            encoder, head, loss_fn, augmentor, dev_ds,
            device, cfg, is_distributed, rank, world_size,
        )
        elapsed = time.time() - t0

        if rank == 0:
            print(f"[Epoch {epoch:3d}/{cfg.epochs}] "
                  f"train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}  "
                  f"time={elapsed:.1f}s")

        # Checkpoint
        if rank == 0:
            _save_ckpt(last_path, epoch, encoder, head, train_loss, dev_loss, cfg)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                patience_counter = 0
                _save_ckpt(best_path, epoch, encoder, head, train_loss, dev_loss, cfg)
                print(f"  → New best checkpoint saved (dev_loss={best_dev_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"  → Early stopping after {cfg.patience} epochs without improvement.")
                    break

    if rank == 0:
        print(f"\nTraining complete. Best checkpoint: {best_path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
