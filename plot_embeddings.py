"""
plot_embeddings.py — UMAP / t-SNE visualisation of Stage 1 embeddings.

Supports two datasets controlled by --dataset:
  asv19  — ASVspoof 2019 LA eval set, coloured by attack type (A01-A19 + Real)
  itw    — In-the-Wild, coloured by Real vs Spoof, hover shows speaker/source

Each dataset produces:
  <plots_dir>/<dataset>/<run_tag>/stage1_<method>_<dataset>.png   (static, 300 dpi)
  <plots_dir>/<dataset>/<run_tag>/stage1_<method>_<dataset>.html  (interactive Plotly)

Usage examples:
    # ASV19 with UMAP
    python plot_embeddings.py --dataset asv19 --ckpt_path /path/to/ntxent_best.pt

    # ITW with t-SNE, save to custom dir
    python plot_embeddings.py --dataset itw --ckpt_path /path/to/ntxent_best.pt \\
        --dr_method tsne --plots_dir ./my_plots

    # Both datasets in one go (run script twice or chain with &&)
    python plot_embeddings.py --dataset asv19 --ckpt_path /path/to/best.pt &&
    python plot_embeddings.py --dataset itw   --ckpt_path /path/to/best.pt
"""

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap

from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from data_loader import (
    ASVspoof2019Dataset,
    InTheWildDataset,
)
from collate import (
    pad_collate_fn_speaker_source_multiclass,
    pad_collate_fn_speaker_source,
)

# ---------------------------------------------------------------------------
# Defaults (override via CLI)
# ---------------------------------------------------------------------------

ASV19_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
ASV19_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1
MAX_DURATION_SECONDS = 10
TARGET_SAMPLE_RATE = 16000
BATCH_SIZE = 64
NUM_WORKERS = 4

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 1337

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_models(ckpt_path: str, model_name: str, device):
    """Load encoder + head from a Stage 1 checkpoint (frozen, eval mode)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("config", {})

    hidden_dim   = cfg.get("hidden_dim",    HIDDEN_DIM)
    input_dim    = cfg.get("input_dim",     INPUT_DIM)
    dropout      = cfg.get("dropout",       DROPOUT)
    use_bottleneck = cfg.get("use_bottleneck", 0)

    encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)
    head    = CompressionModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout,
        use_bottleneck=bool(use_bottleneck),
    ).to(device)

    if "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
        # strip DDP "module." prefix if present
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                 for k, v in state.items()}
        encoder.load_state_dict(state, strict=True)
        print("  Loaded encoder weights from checkpoint.")

    head_state = ckpt.get("compression_state_dict", ckpt)
    head_state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in head_state.items()}
    head.load_state_dict(head_state, strict=True)

    encoder.eval()
    head.eval()
    return encoder, head


@torch.no_grad()
def extract_embeddings_asv19(encoder, head, loader, device):
    """Returns (embs, bin_labels, attack_ids, filenames) numpy arrays."""
    all_embs, all_bin, all_atk, all_names = [], [], [], []
    for batch in loader:
        waveforms, bin_labels, attack_ids, _, sources = batch
        waveforms = waveforms.to(device)
        attn = (waveforms != 0.0).long()
        hs  = encoder(waveforms, attention_mask=attn)
        seq = head(hs)
        z   = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        all_embs.append(z.cpu().numpy())
        all_bin.append(bin_labels.numpy())
        all_atk.append(attack_ids.numpy())
        all_names.extend(list(sources))
    return (
        np.concatenate(all_embs),
        np.concatenate(all_bin),
        np.concatenate(all_atk),
        all_names,
    )


@torch.no_grad()
def extract_embeddings_itw(encoder, head, loader, device):
    """Returns (embs, bin_labels, speakers, sources) numpy / list."""
    all_embs, all_bin, all_spk, all_src = [], [], [], []
    for batch in loader:
        waveforms, bin_labels, speakers, sources = batch
        waveforms = waveforms.to(device)
        attn = (waveforms != 0.0).long()
        hs  = encoder(waveforms, attention_mask=attn)
        seq = head(hs)
        z   = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        all_embs.append(z.cpu().numpy())
        all_bin.append(bin_labels.numpy())
        all_spk.extend([str(s) for s in speakers])
        all_src.extend([str(s) for s in sources])
    return (
        np.concatenate(all_embs),
        np.concatenate(all_bin),
        all_spk,
        all_src,
    )


def reduce_dims(embs: np.ndarray, method: str) -> np.ndarray:
    if method == "umap":
        print(f"  Running UMAP on {embs.shape[0]} samples ...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=RANDOM_STATE,
        )
    else:
        print(f"  Running t-SNE on {embs.shape[0]} samples ...")
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=RANDOM_STATE,
        )
    return reducer.fit_transform(embs)


def save_png_asv19(embs_2d, attack_labels, method, plots_dir, exp_name):
    """Static PNG coloured by attack type. Real=blue, attacks get auto colours."""
    plt.figure(figsize=(10, 8))
    unique = sorted(set(attack_labels))

    if "Real" in unique:
        m = attack_labels == "Real"
        plt.scatter(embs_2d[m, 0], embs_2d[m, 1], s=8, alpha=0.6, c="royalblue", label="Real")

    for lab in unique:
        if lab == "Real":
            continue
        m = attack_labels == lab
        if m.any():
            plt.scatter(embs_2d[m, 0], embs_2d[m, 1], s=8, alpha=0.6, label=lab)

    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.title(f"{method.upper()} — ASV19 eval  [{exp_name}]")
    plt.xlabel(f"{method.upper()}-1")
    plt.ylabel(f"{method.upper()}-2")
    plt.tight_layout()

    path = os.path.join(plots_dir, f"stage1_{method}_asv19.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved: {path}")
    return path


def save_html_asv19(embs_2d, attack_labels, filenames, method, plots_dir, exp_name):
    fig = px.scatter(
        x=embs_2d[:, 0], y=embs_2d[:, 1],
        color=attack_labels,
        hover_name=filenames,
        title=f"{method.upper()} — ASV19 eval [{exp_name}]",
        labels={"x": f"{method.upper()}-1", "y": f"{method.upper()}-2", "color": "Class"},
        color_discrete_map={"Real": "royalblue"},
    )
    path = os.path.join(plots_dir, f"stage1_{method}_asv19.html")
    fig.write_html(path)
    print(f"  HTML saved: {path}")


def save_png_itw(embs_2d, class_labels, method, plots_dir, exp_name):
    """Static PNG: Real=blue, Spoof=red."""
    fig, ax = plt.subplots(figsize=(10, 8))
    m_real  = class_labels == "Real"
    m_spoof = class_labels == "Spoof"
    if m_real.any():
        ax.scatter(embs_2d[m_real,  0], embs_2d[m_real,  1], s=8, alpha=0.6, c="royalblue", label="Real")
    if m_spoof.any():
        ax.scatter(embs_2d[m_spoof, 0], embs_2d[m_spoof, 1], s=8, alpha=0.6, c="crimson",   label="Spoof")
    ax.legend(markerscale=2, fontsize=10)
    ax.set_title(f"{method.upper()} — In-the-Wild  [{exp_name}]")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"stage1_{method}_itw.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved: {path}")


def save_html_itw(embs_2d, class_labels, speakers, sources, method, plots_dir, exp_name):
    df = pd.DataFrame({
        f"{method.upper()}-1": embs_2d[:, 0],
        f"{method.upper()}-2": embs_2d[:, 1],
        "Class":   class_labels,
        "Speaker": speakers,
        "Source":  sources,
    })
    fig = px.scatter(
        df,
        x=f"{method.upper()}-1", y=f"{method.upper()}-2",
        color="Class",
        hover_data=["Speaker", "Source"],
        title=f"{method.upper()} — In-the-Wild [{exp_name}]",
        color_discrete_map={"Real": "royalblue", "Spoof": "crimson"},
    )
    path = os.path.join(plots_dir, f"stage1_{method}_itw.html")
    fig.write_html(path)
    print(f"  HTML saved: {path}")


# ---------------------------------------------------------------------------
# Dataset-specific runners
# ---------------------------------------------------------------------------

def run_asv19(args, encoder, head, run_tag, plots_dir):
    print("\n[ASV19] Loading eval dataset ...")
    ds = ASVspoof2019Dataset(
        root_dir=args.asv19_eval_root,
        protocol_file=args.asv19_eval_protocol,
        subset="all",
        max_duration_seconds=args.max_duration_seconds,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
    )
    idx_to_attack = {v: k for k, v in ds.attack_to_idx.items()}

    print(f"[ASV19] Extracting embeddings ({len(ds)} utterances) ...")
    embs, bin_labels, atk_ids, filenames = extract_embeddings_asv19(
        encoder, head, loader, DEVICE
    )
    print(f"[ASV19] {embs.shape[0]} embeddings extracted.")

    attack_labels = np.array([
        "Real" if b == 1 else idx_to_attack.get(int(a), f"A{int(a):02d}")
        for b, a in zip(bin_labels, atk_ids)
    ])

    embs_2d = reduce_dims(embs, args.dr_method)

    save_png_asv19(embs_2d, attack_labels, args.dr_method, plots_dir, args.exp_name)
    save_html_asv19(embs_2d, attack_labels, filenames, args.dr_method, plots_dir, args.exp_name)


def run_itw(args, encoder, head, run_tag, plots_dir):
    print("\n[ITW] Loading dataset ...")
    ds = InTheWildDataset(
        root_dir=args.itw_root,
        protocol_file=args.itw_protocol,
        subset=None,
        max_duration_seconds=args.max_duration_seconds,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source,
    )

    print(f"[ITW] Extracting embeddings ({len(ds)} utterances) ...")
    embs, bin_labels, speakers, sources = extract_embeddings_itw(
        encoder, head, loader, DEVICE
    )
    print(f"[ITW] {embs.shape[0]} embeddings extracted.")

    class_labels = np.array(["Real" if int(b) == 1 else "Spoof" for b in bin_labels])

    embs_2d = reduce_dims(embs, args.dr_method)

    save_png_itw(embs_2d, class_labels, args.dr_method, plots_dir, args.exp_name)
    save_html_itw(embs_2d, class_labels, speakers, sources, args.dr_method, plots_dir, args.exp_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Stage 1 embeddings with UMAP or t-SNE."
    )

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["asv19", "itw", "both"],
                        help="Which dataset to plot. 'both' runs asv19 then itw.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to Stage 1 checkpoint (.pt).")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment label shown in plot titles.")

    # Dimensionality reduction
    parser.add_argument("--dr_method", type=str, default="umap",
                        choices=["umap", "tsne"])

    # ASV19 paths
    parser.add_argument("--asv19_eval_root",     type=str, default=ASV19_EVAL_ROOT)
    parser.add_argument("--asv19_eval_protocol", type=str, default=ASV19_EVAL_PROTOCOL)

    # ITW paths
    parser.add_argument("--itw_root",     type=str, default=ITW_ROOT)
    parser.add_argument("--itw_protocol", type=str, default=ITW_PROTOCOL)

    # Output
    parser.add_argument("--plots_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "plots"))

    # Audio / loader
    parser.add_argument("--max_duration_seconds", type=float, default=MAX_DURATION_SECONDS)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)

    args = parser.parse_args()
    if not args.exp_name:
        args.exp_name = os.path.basename(os.path.dirname(args.ckpt_path))

    set_seed(RANDOM_STATE)

    run_tag   = args.model_name.replace("/", "__")
    plots_dir = os.path.join(args.plots_dir, run_tag)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Device   : {DEVICE}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Model    : {args.model_name}")
    print(f"Method   : {args.dr_method.upper()}")
    print(f"Output   : {plots_dir}")

    print("\nLoading models ...")
    encoder, head = load_models(args.ckpt_path, args.model_name, DEVICE)

    datasets = ["asv19", "itw"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        if ds == "asv19":
            run_asv19(args, encoder, head, run_tag, plots_dir)
        else:
            run_itw(args, encoder, head, run_tag, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
