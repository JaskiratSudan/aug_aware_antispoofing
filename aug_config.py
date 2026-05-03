"""
aug_config.py — Argument parser for aug_aware_antispoofing training scripts.

Mirrors the pattern from wav2vec_contr_loss/stage1_config.py but strips
label-dependent fields (supcon_similarity, uniformity_weight, topk_neg, etc.)
and adds NT-Xent / augmentation-specific arguments.
"""

import argparse
import os

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"
DEV_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac"
DEV_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt"

_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

EPOCHS = 100
BATCH_SIZE = 32
NUM_SAMPLES = None
HEAD_LR = 5e-4
ENC_LR = 1e-5
WEIGHT_DECAY = 3e-3
NUM_WORKERS = 4
SEED = 1337
MAX_DURATION_SECONDS = 10

SAVE_DIR = os.path.join(_HERE, "checkpoints_stage1")

# NT-Xent / augmentation
N_VIEWS = 5
TEMPERATURE = 0.2
NOISE_DIR = os.path.join(_HERE, "bg_noise")
RIR_DIR = ""
AUG_MODE = "chain"
FREEZE_ENCODER = 0  # 0 = fine-tune encoder (default), 1 = freeze


def build_config():
    parser = argparse.ArgumentParser(description="Stage 1 NT-Xent training with augmentation views")

    # Data
    parser.add_argument("--train_root",     type=str, default=TRAIN_ROOT)
    parser.add_argument("--train_protocol", type=str, default=TRAIN_PROTOCOL)
    parser.add_argument("--dev_root",       type=str, default=DEV_ROOT)
    parser.add_argument("--dev_protocol",   type=str, default=DEV_PROTOCOL)
    parser.add_argument("--num_samples",    type=str, default=str(NUM_SAMPLES),
                        help="Max training samples ('none' for all).")
    parser.add_argument("--max_duration_seconds", type=float, default=MAX_DURATION_SECONDS)

    # Model
    parser.add_argument("--model_name",      type=str,   default=MODEL_NAME)
    parser.add_argument("--input_dim",       type=int,   default=INPUT_DIM)
    parser.add_argument("--hidden_dim",      type=int,   default=HIDDEN_DIM)
    parser.add_argument("--dropout",         type=float, default=DROPOUT)
    parser.add_argument("--freeze_encoder",  type=int,   default=FREEZE_ENCODER,
                        help="0=fine-tune encoder (default), 1=freeze encoder.")
    parser.add_argument("--use_bottleneck",  type=int,   default=0)

    # Training
    parser.add_argument("--save_dir",      type=str,   default=SAVE_DIR)
    parser.add_argument("--epochs",        type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--enc_lr",        type=float, default=ENC_LR)
    parser.add_argument("--head_lr",       type=float, default=HEAD_LR)
    parser.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num_workers",   type=int,   default=NUM_WORKERS)
    parser.add_argument("--seed",          type=int,   default=SEED)
    parser.add_argument("--patience",      type=int,   default=10,
                        help="Early-stopping patience (epochs without dev loss improvement).")
    parser.add_argument("--accum_steps",   type=int,   default=1,
                        help="Gradient accumulation steps.")

    # NT-Xent / augmentation
    parser.add_argument("--n_views",      type=int,   default=N_VIEWS,
                        help="Total augmented views per utterance (no +1 offset).")
    parser.add_argument("--temperature",  type=float, default=TEMPERATURE,
                        help="NT-Xent temperature (default 0.2, same scale as SupCon).")
    parser.add_argument("--noise_dir",    type=str,   default=NOISE_DIR,
                        help="Path to background noise directory. Empty = Cat1-only fallback.")
    parser.add_argument("--rir_dir",      type=str,   default=RIR_DIR,
                        help="Path to RIR directory for AddReverb. Empty = skip reverb.")
    parser.add_argument("--aug_mode",     type=str,   default=AUG_MODE,
                        choices=["single", "chain"],
                        help="'single': one random aug; 'chain': 1-2 Cat1 + optional Cat2.")

    # Training data subset
    parser.add_argument("--train_bonafide_only", type=int, default=0,
                        help="1 = train on bonafide-only samples (one-class NT-Xent). "
                             "Faster and better for generalisation — spoof samples add no "
                             "useful positives to a self-supervised objective.")

    # Resume
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from.")

    cfg = parser.parse_args()

    # Normalise num_samples
    if isinstance(cfg.num_samples, str) and cfg.num_samples.lower() == "none":
        cfg.num_samples = None
    elif cfg.num_samples is not None:
        try:
            cfg.num_samples = int(cfg.num_samples)
        except (ValueError, TypeError):
            cfg.num_samples = None

    return cfg
