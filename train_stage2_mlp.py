"""
train_stage2_mlp.py — Stage 2 MLP classifier training for aug_aware_antispoofing.

Loads a frozen Stage 1 checkpoint (encoder + projection head), extracts 256-dim
L2-normalized embeddings on-the-fly, and trains MLPClassifier (256→64→1) with
BCEBinaryLoss.

Usage:
    python train_stage2_mlp.py \\
        --stage1_ckpt /path/to/ntxent_best.pt \\
        --model_name facebook/wav2vec2-xls-r-300m \\
        [--train_root ...] [--train_protocol ...]
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from asvspoof_windowed_loader import ASVspoof2019WindowedDataset
from collate import pad_collate_fn_speaker_source_multiclass
from stage1_utils import set_seed

from classifier import MLPClassifier
from losses import BCEBinaryLoss

# ---------------------------------------------------------------------------
# Defaults (mirrors aug_config paths)
# ---------------------------------------------------------------------------

TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"
DEV_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac"
DEV_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt"

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_config():
    parser = argparse.ArgumentParser(description="Stage 2 MLP classifier training")

    parser.add_argument("--stage1_ckpt",  type=str, required=True,
                        help="Path to Stage 1 NT-Xent best checkpoint.")
    parser.add_argument("--model_name",   type=str,
                        default="facebook/wav2vec2-xls-r-300m")

    parser.add_argument("--train_root",     type=str, default=TRAIN_ROOT)
    parser.add_argument("--train_protocol", type=str, default=TRAIN_PROTOCOL)
    parser.add_argument("--dev_root",       type=str, default=DEV_ROOT)
    parser.add_argument("--dev_protocol",   type=str, default=DEV_PROTOCOL)

    parser.add_argument("--save_dir",      type=str,
                        default=os.path.join(_HERE, "checkpoints_stage2"))
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--dropout",       type=float, default=0.2)
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--seed",          type=int,   default=1337)
    parser.add_argument("--max_duration_seconds", type=float, default=10.0)
    parser.add_argument("--use_bottleneck",type=int, default=0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(encoder, head, loader, device):
    """
    Run encoder + head over the dataset and return (embeddings, labels).
    embeddings: (N, 256) L2-normalized  labels: (N,) int
    """
    encoder.eval()
    head.eval()
    all_z, all_labels = [], []

    for batch in loader:
        waveforms, bin_labels = batch[0].to(device), batch[1]
        attn_mask = (waveforms != 0.0).long()
        hs  = encoder(waveforms, attention_mask=attn_mask)   # (B, K, F, T)
        seq = head(hs)                                        # (B, 256, T)
        z   = F.normalize(seq.mean(dim=-1), p=2, dim=1)      # (B, 256)
        all_z.append(z.cpu())
        all_labels.append(bin_labels)

    return torch.cat(all_z, dim=0), torch.cat(all_labels, dim=0).float()


# ---------------------------------------------------------------------------
# Train / eval on cached embeddings
# ---------------------------------------------------------------------------

def train_epoch(classifier, optimizer, loss_fn, z_all, y_all, batch_size, device):
    classifier.train()
    N = z_all.shape[0]
    perm = torch.randperm(N)
    total_loss, steps = 0.0, 0

    for i in range(0, N, batch_size):
        idx = perm[i : i + batch_size]
        z = z_all[idx].to(device)
        y = y_all[idx].to(device)
        logits = classifier(z)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def eval_epoch(classifier, loss_fn, z_all, y_all, batch_size, device):
    classifier.eval()
    N = z_all.shape[0]
    total_loss, steps = 0.0, 0

    for i in range(0, N, batch_size):
        z = z_all[i : i + batch_size].to(device)
        y = y_all[i : i + batch_size].to(device)
        logits = classifier(z)
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = build_config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2] device={device}  ckpt={cfg.stage1_ckpt}")

    # ------------------------------------------------------------------
    # Load Stage 1 models (frozen)
    # ------------------------------------------------------------------
    ckpt = torch.load(cfg.stage1_ckpt, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    model_name  = ckpt_cfg.get("model_name",   cfg.model_name)
    hidden_dim  = ckpt_cfg.get("hidden_dim",   256)
    input_dim   = ckpt_cfg.get("input_dim",    1024)
    dropout_enc = ckpt_cfg.get("dropout",      0.1)
    use_bn      = ckpt_cfg.get("use_bottleneck", 0)

    encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)
    head    = CompressionModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_enc,
        use_bottleneck=bool(use_bn),
    ).to(device)

    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print("[Stage 2] Loaded encoder from Stage 1 checkpoint.")
    head.load_state_dict(ckpt["compression_state_dict"])
    print("[Stage 2] Loaded projection head from Stage 1 checkpoint.")

    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    train_ds = ASVspoof2019WindowedDataset(
        protocol_file=cfg.train_protocol,
        root_dir=cfg.train_root,
        subset="all",
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
    print(f"[Stage 2] train={len(train_ds)}  dev={len(dev_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Pre-extract embeddings (one pass, then train from cache)
    # ------------------------------------------------------------------
    print("[Stage 2] Extracting train embeddings ...")
    t0 = time.time()
    z_train, y_train = extract_embeddings(encoder, head, train_loader, device)
    print(f"  done in {time.time()-t0:.1f}s  shape={z_train.shape}")

    print("[Stage 2] Extracting dev embeddings ...")
    t0 = time.time()
    z_dev, y_dev = extract_embeddings(encoder, head, dev_loader, device)
    print(f"  done in {time.time()-t0:.1f}s  shape={z_dev.shape}")

    # Move embedding cache to CPU for DataLoader-style batching in training loop
    z_train, y_train = z_train.cpu(), y_train.cpu()
    z_dev,   y_dev   = z_dev.cpu(),   y_dev.cpu()

    # ------------------------------------------------------------------
    # MLPClassifier
    # ------------------------------------------------------------------
    pos_count = (y_train == 1).sum().item()
    neg_count = (y_train == 0).sum().item()
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], device=device)

    classifier = MLPClassifier(
        input_dim=hidden_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    loss_fn   = BCEBinaryLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model_tag  = cfg.stage1_ckpt.replace("/", "__").replace(".", "_")
    save_root  = os.path.join(cfg.save_dir, os.path.basename(os.path.dirname(cfg.stage1_ckpt)))
    best_path  = os.path.join(save_root, "stage2_mlp_best.pt")
    os.makedirs(save_root, exist_ok=True)

    best_dev_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(
            classifier, optimizer, loss_fn, z_train, y_train, cfg.batch_size, device
        )
        dev_loss = eval_epoch(
            classifier, loss_fn, z_dev, y_dev, cfg.batch_size, device
        )

        print(f"[Epoch {epoch:3d}/{cfg.epochs}] "
              f"train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "state_dict": classifier.state_dict(),
                "dev_loss": dev_loss,
                "train_loss": train_loss,
                "config": vars(cfg),
            }, best_path)
            print(f"  → New best saved (dev_loss={best_dev_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"  → Early stopping.")
                break

    print(f"\nStage 2 complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
