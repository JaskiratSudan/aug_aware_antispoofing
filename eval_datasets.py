"""
eval_datasets.py — Score and EER evaluation for aug_aware_antispoofing.

Adapted from wav2vec_contr_loss/eval_datasets.py.
Key differences:
  - Stage 1 checkpoint config keys are lowercase (hidden_dim, input_dim, dropout)
  - Stage 2 checkpoint uses 'state_dict' key and loads MLPClassifier
  - Handles both old (uppercase) and new (lowercase) config key formats
"""

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from data_loader import (
    ASVspoof2019Dataset,
    ASVspoof5Dataset,
    ASVspoof2021DFDataset,
    ASVspoof2021LADataset,
    InTheWildDataset,
    FamousFiguresDataset,
    FakeXposeDataset,
    MLAADMailabsDataset,
    DeepfakeEval2024Dataset,
)
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from classifier import MLPClassifier
from evaluation import calculate_EER

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_ASV19_EVAL_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
DEFAULT_ASV19_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

DEFAULT_ASV5_EVAL_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac"
DEFAULT_ASV5_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv"

DEFAULT_ASV21_DF_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/flac"
DEFAULT_ASV21_DF_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/trial_metadata.txt"

DEFAULT_ASV21_LA_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/flac"
DEFAULT_ASV21_LA_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/trial_metadata.txt"

DEFAULT_ITW_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
DEFAULT_ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

DEFAULT_FF_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt"
DEFAULT_FF_ROOT     = ""

DEFAULT_FAKEXPOSE_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/fakexpose"

DEFAULT_MLAAD_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/multilingual"
DEFAULT_MLAAD_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt"

DEFAULT_DEEPFAKE_EVAL_ROOT     = "/nfs/turbo/umd-hafiz/issf_server_data/Deepfake_Eval_2024/audio-data"
DEFAULT_DEEPFAKE_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/Deepfake_Eval_2024/audio-metadata-publish.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_dict_flexible(model: nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=True)


def _cfg_get(cfg: dict, *keys, default=None):
    """Try multiple key names (lowercase then UPPERCASE) to handle both formats."""
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class Stage1Backbone(nn.Module):
    """Frozen encoder + projection head from a Stage 1 NT-Xent checkpoint."""

    def __init__(self, ckpt_path: str, model_name: str, device: torch.device):
        super().__init__()
        ckpt = safe_load(ckpt_path, device)
        cfg  = ckpt.get("config", {})

        # Support both lowercase (new) and UPPERCASE (old) config keys
        input_dim  = _cfg_get(cfg, "input_dim",  "INPUT_DIM",  default=1024)
        hidden_dim = _cfg_get(cfg, "hidden_dim", "HIDDEN_DIM", default=256)
        dropout    = _cfg_get(cfg, "dropout",    "DROPOUT",    default=0.1)
        use_bn     = _cfg_get(cfg, "use_bottleneck", "USE_BOTTLENECK", default=0)

        self.encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)
        if "encoder_state_dict" in ckpt:
            load_state_dict_flexible(self.encoder, ckpt["encoder_state_dict"])
            print(f"  [Stage1] Loaded fine-tuned encoder from {os.path.basename(ckpt_path)}")

        self.head = CompressionModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            use_bottleneck=bool(use_bn),
        ).to(device)
        load_state_dict_flexible(self.head, ckpt["compression_state_dict"])

        self.encoder.eval()
        self.head.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs  = self.encoder(waveforms, attention_mask=attention_mask)
        seq = self.head(hs)
        z   = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        return z


def load_stage2_clf(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load Stage 2 classifier.  Handles:
      - New format: MLPClassifier saved by train_stage2_mlp.py
          keys: 'state_dict', config with lowercase 'hidden_dim'/'dropout'
      - Old format: linear/mlp/deep_mlp heads saved by train_stage2_classifier.py
          keys: 'model_state_dict', config with uppercase 'HEAD_TYPE' etc.
    """
    ckpt = safe_load(ckpt_path, device)
    cfg  = ckpt.get("config", {})

    # ---- New format (MLPClassifier) ----
    if "state_dict" in ckpt:
        hidden_dim = _cfg_get(cfg, "hidden_dim", "HIDDEN_DIM", default=64)
        dropout    = _cfg_get(cfg, "dropout",    "DROPOUT",    default=0.2)
        clf = MLPClassifier(input_dim=256, hidden_dim=hidden_dim, dropout=dropout).to(device)
        load_state_dict_flexible(clf, ckpt["state_dict"])
        clf.eval()
        for p in clf.parameters():
            p.requires_grad_(False)
        print(f"  [Stage2] MLPClassifier  hidden={hidden_dim}  dropout={dropout}")
        return clf

    # ---- Old format (linear / mlp / deep_mlp from wav2vec_contr_loss) ----
    head_type  = _cfg_get(cfg, "HEAD_TYPE", default="linear")
    in_dim     = _cfg_get(cfg, "IN_DIM",    default=256)
    hidden_dim = _cfg_get(cfg, "HIDDEN_DIM", default=128)
    dropout    = _cfg_get(cfg, "DROPOUT",   default=0.2)

    if head_type == "linear":
        clf = nn.Linear(in_dim, 1).to(device)
        # wrap so .forward returns (B,) not (B,1)
        clf = _LinearWrapper(clf).to(device)
    elif head_type in ("mlp", "small_mlp"):
        clf = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        clf = _SeqWrapper(clf).to(device)
    else:
        raise ValueError(f"Unsupported HEAD_TYPE in old checkpoint: {head_type}")

    load_state_dict_flexible(clf, ckpt["model_state_dict"])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    print(f"  [Stage2] Old-format head: type={head_type}")
    return clf


class _LinearWrapper(nn.Module):
    def __init__(self, fc): super().__init__(); self.fc = fc
    def forward(self, x): return self.fc(x).squeeze(-1)


class _SeqWrapper(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def pad_collate_fn_generic(batch):
    waveforms = [item[0] for item in batch]
    labels    = [item[1] for item in batch]
    sources, utt_ids = [], []
    for item in batch:
        if len(item) >= 4:
            sources.append(item[-2])
            utt_ids.append(item[-1])
        elif len(item) == 3:
            sources.append("NA")
            utt_ids.append(item[2])
        else:
            sources.append("NA")
            utt_ids.append("unknown")
    padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    return padded, torch.stack(labels), None, sources, utt_ids


@torch.no_grad()
def score_and_write(stage1, stage2, loader, device, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stage1.eval(); stage2.eval()
    with open(out_path, "w") as f:
        for batch in loader:
            waveforms, labels, _, sources, utt_ids = batch
            waveforms = waveforms.to(device)
            attn      = (waveforms != 0.0).long()
            embs      = stage1(waveforms, attn)
            logits    = stage2(embs)
            scores    = logits.detach().cpu().numpy()
            labs_np   = labels.numpy().astype(int)
            for i in range(len(scores)):
                utt_id = str(utt_ids[i]).strip().replace(" ", "_")
                src    = str(sources[i]).strip().replace(" ", "_")
                key    = "bonafide" if labs_np[i] == 1 else "spoof"
                f.write(f"{utt_id} {src} {key} {scores[i]:.6f}\n")
    print(f"  Wrote: {out_path}")


def _merge_ranked(out_path, world_size):
    with open(out_path, "w") as fout:
        for rank in range(world_size):
            part = f"{out_path}.rank{rank}"
            if os.path.isfile(part):
                with open(part) as fin:
                    shutil.copyfileobj(fin, fout)
                os.remove(part)


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------

def dataset_specs(args):
    ff_speakers = [s.strip() for s in args.ff_speakers.split(",") if s.strip()]
    return {
        "asv19": {
            "cls": ASVspoof2019Dataset,
            "kwargs": dict(
                root_dir=args.asv19_root, protocol_file=args.asv19_protocol,
                subset=args.subset, num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate,
            ),
            "score_rel": os.path.join(args.model_name, "score_cm_eval.txt"),
        },
        "asv5": {
            "cls": ASVspoof5Dataset,
            "kwargs": dict(
                root_dir=args.asv5_root, protocol_file=args.asv5_protocol,
                subset=args.subset, num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate,
            ),
            "score_rel": os.path.join(args.model_name, "score_cm_eval_asv5.txt"),
        },
        "asv21_df": {
            "cls": ASVspoof2021DFDataset,
            "kwargs": dict(
                root_dir=args.asv21_df_root, protocol_file=args.asv21_df_protocol,
                subset=args.subset, num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate, skip_missing=True,
            ),
            "score_rel": os.path.join("asv21_df", "score_cm_asv21_df.txt"),
        },
        "asv21_la": {
            "cls": ASVspoof2021LADataset,
            "kwargs": dict(
                root_dir=args.asv21_la_root, protocol_file=args.asv21_la_protocol,
                subset=args.subset, num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate, skip_missing=True,
            ),
            "score_rel": os.path.join("asv21_la", "score_cm_asv21_la.txt"),
        },
        "itw": {
            "cls": InTheWildDataset,
            "kwargs": dict(
                root_dir=args.itw_root, protocol_file=args.itw_protocol,
                subset=args.subset, num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
            ),
            "score_rel": os.path.join(args.model_name, "score_cm_itw.txt"),
        },
        "famous_figures": {
            "cls": FamousFiguresDataset,
            "kwargs": dict(
                protocol_file=args.ff_protocol, root_dir=args.ff_root,
                subset=args.subset, num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate,
                include_speakers=ff_speakers if ff_speakers else None,
                return_audio_name=True,
            ),
            "score_rel": os.path.join("famous_figures", "score_cm_ff.txt"),
        },
        "fakexpose": {
            "cls": FakeXposeDataset,
            "kwargs": dict(
                root_dir=args.fakexpose_root, subset=args.subset,
                num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate,
            ),
            "score_rel": os.path.join("fakexpose", "score_cm_fakexpose.txt"),
        },
        "mlaad": {
            "cls": MLAADMailabsDataset,
            "kwargs": dict(
                root_dir=args.mlaad_root, protocol_file=args.mlaad_protocol,
                subset="all", num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate,
            ),
            "score_rel": os.path.join("mlaad", "score_cm_mlaad.txt"),
        },
        "deepfake_eval_2024": {
            "cls": DeepfakeEval2024Dataset,
            "kwargs": dict(
                root_dir=args.deepfake_root, protocol_file=args.deepfake_protocol,
                subset=args.subset, num_samples=args.num_samples,
                max_duration_seconds=args.max_duration_seconds,
                target_sample_rate=args.target_sample_rate,
            ),
            "score_rel": os.path.join("deepfake_eval_2024", "score_cm_deepfake_eval_2024.txt"),
        },
    }


def _missing_inputs(kwargs):
    missing = []
    proto = kwargs.get("protocol_file")
    if proto and not os.path.isfile(proto):
        missing.append(proto)
    root = kwargs.get("root_dir")
    if root and not os.path.isdir(root):
        missing.append(root)
    return missing


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank        = int(os.environ["RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])
        local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score datasets and compute EER.")
    ap.add_argument("--exp_name",    type=str, default="")
    ap.add_argument("--datasets",    type=str, default="asv19,itw,fakexpose",
                    help="Comma-separated list of datasets to evaluate.")
    ap.add_argument("--stage1_ckpt", type=str, required=True)
    ap.add_argument("--stage2_ckpt", type=str, required=True)
    ap.add_argument("--scores_dir",  type=str, default="scores")
    ap.add_argument("--model_name",  type=str, default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--subset",      type=str, default="all",
                    choices=["all", "bonafide", "spoof"])
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_duration_seconds", type=int,   default=10)
    ap.add_argument("--target_sample_rate",   type=int,   default=16000)
    ap.add_argument("--num_samples",          type=int,   default=None)
    ap.add_argument("--print_eer",   action="store_true")
    ap.add_argument("--force_rescore", action="store_true",
                    help="Re-score even if the score file already exists.")

    # Per-dataset path overrides
    ap.add_argument("--asv19_root",      type=str, default=DEFAULT_ASV19_EVAL_ROOT)
    ap.add_argument("--asv19_protocol",  type=str, default=DEFAULT_ASV19_EVAL_PROTOCOL)
    ap.add_argument("--asv5_root",       type=str, default=DEFAULT_ASV5_EVAL_ROOT)
    ap.add_argument("--asv5_protocol",   type=str, default=DEFAULT_ASV5_EVAL_PROTOCOL)
    ap.add_argument("--asv21_df_root",   type=str, default=DEFAULT_ASV21_DF_ROOT)
    ap.add_argument("--asv21_df_protocol", type=str, default=DEFAULT_ASV21_DF_PROTOCOL)
    ap.add_argument("--asv21_la_root",   type=str, default=DEFAULT_ASV21_LA_ROOT)
    ap.add_argument("--asv21_la_protocol", type=str, default=DEFAULT_ASV21_LA_PROTOCOL)
    ap.add_argument("--itw_root",        type=str, default=DEFAULT_ITW_ROOT)
    ap.add_argument("--itw_protocol",    type=str, default=DEFAULT_ITW_PROTOCOL)
    ap.add_argument("--ff_protocol",     type=str, default=DEFAULT_FF_PROTOCOL)
    ap.add_argument("--ff_root",         type=str, default=DEFAULT_FF_ROOT)
    ap.add_argument("--ff_speakers",     type=str, default="")
    ap.add_argument("--fakexpose_root",  type=str, default=DEFAULT_FAKEXPOSE_ROOT)
    ap.add_argument("--mlaad_root",      type=str, default=DEFAULT_MLAAD_ROOT)
    ap.add_argument("--mlaad_protocol",  type=str, default=DEFAULT_MLAAD_PROTOCOL)
    ap.add_argument("--deepfake_root",   type=str, default=DEFAULT_DEEPFAKE_EVAL_ROOT)
    ap.add_argument("--deepfake_protocol", type=str, default=DEFAULT_DEEPFAKE_EVAL_PROTOCOL)

    args = ap.parse_args()

    is_distributed, rank, world_size, local_rank = setup_distributed()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank) if is_distributed else torch.device("cuda")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Device     : {device}")
        print(f"Stage1 ckpt: {args.stage1_ckpt}")
        print(f"Stage2 ckpt: {args.stage2_ckpt}")
        print(f"Datasets   : {args.datasets}")

    stage1 = Stage1Backbone(args.stage1_ckpt, model_name=args.model_name, device=device)
    stage2 = load_stage2_clf(args.stage2_ckpt, device=device)

    if not is_distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        stage1 = nn.DataParallel(stage1)
        stage2 = nn.DataParallel(stage2)

    spec_map     = dataset_specs(args)
    score_exp    = args.exp_name or "custom"
    dataset_list = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    for name in dataset_list:
        if name not in spec_map:
            print(f"[WARN] Unknown dataset '{name}', skipping.")
            continue
        spec    = spec_map[name]
        missing = _missing_inputs(spec["kwargs"])
        if missing:
            print(f"[WARN] Missing inputs for '{name}': {missing}. Skipping.")
            continue

        score_path = os.path.join(args.scores_dir, score_exp, spec["score_rel"])
        if os.path.isfile(score_path) and not args.force_rescore:
            if rank == 0:
                print(f"[SKIP] Score file exists: {score_path}  (use --force_rescore to overwrite)")
        else:
            if rank == 0:
                print(f"\n[{name.upper()}] Scoring -> {score_path}")

            ds = spec["cls"](**spec["kwargs"])

            if is_distributed:
                sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
                loader  = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                                     num_workers=args.num_workers, pin_memory=True,
                                     collate_fn=pad_collate_fn_generic)
            else:
                loader  = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True,
                                     collate_fn=pad_collate_fn_generic)

            out_path = f"{score_path}.rank{rank}" if is_distributed else score_path
            score_and_write(stage1, stage2, loader, device, out_path)

            if is_distributed:
                dist.barrier()
                if rank == 0:
                    _merge_ranked(score_path, world_size)
                dist.barrier()

        if args.print_eer and rank == 0:
            if os.path.isfile(score_path):
                eer = calculate_EER(score_path)
                print(f"  EER [{name}]: {eer:.2f}%")
            else:
                print(f"  [WARN] Score file not found for EER: {score_path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
