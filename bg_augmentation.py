"""
bg_augmentation.py

Self-contained background noise augmentation for the wav2vec_contr_loss pipeline.

Background noise files live in:  wav2vec_contr_loss/bg_noise/

Augmentation scheme (mirrors Dataset_safe_train_BG_split from
/data/FF_V2/Speaker_Specific_OCSVM/Mixture/data_utils.py):

  p < 0.20  → RawBoost only
  0.20 ≤ p < 0.40  → background noise only
  0.40 ≤ p < 0.60  → background noise + RawBoost
  p ≥ 0.60  → no augmentation (pass-through)

Public API used by the training pipelines:
  apply_aug_split_batch(x, cfg)   — for stage1 (reads cfg.bg_files, cfg.target_sample_rate)
  apply_aug_split_batch_baseline(x) — for baseline_train (reads module-level _BG_FILES)
"""

import glob
import os
import random

import librosa
import numpy as np
import torch

from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise


_DEFAULT_BG_NOISE_DIR = os.path.join(os.path.dirname(__file__), "bg_noise")


def _discover_bg_files(bg_noise_dir: str = _DEFAULT_BG_NOISE_DIR):
    files = glob.glob(os.path.join(bg_noise_dir, "*.mp3"))
    return files


# ---------------------------------------------------------------------------
# Core DSP helpers
# ---------------------------------------------------------------------------

def mix_mild_background(speech: np.ndarray, sr: int, bg_files: list,
                        snr_min: float = 18.0, snr_max: float = 28.0) -> np.ndarray:
    """Mix speech with a randomly selected background noise file at a mild SNR."""
    if not bg_files:
        return speech
    bg, _ = librosa.load(random.choice(bg_files), sr=sr, mono=True)
    if len(bg) == 0:
        return speech
    if len(bg) < len(speech):
        bg = np.tile(bg, int(np.ceil(len(speech) / len(bg))))
    start = random.randint(0, len(bg) - len(speech))
    bg = bg[start:start + len(speech)]
    bg = bg - np.mean(bg)
    speech_power = np.mean(speech ** 2) + 1e-8
    bg_power = np.mean(bg ** 2) + 1e-8
    snr_db = random.uniform(snr_min, snr_max)
    target_bg_power = speech_power / (10 ** (snr_db / 10))
    bg = bg * np.sqrt(target_bg_power / bg_power)
    mixed = speech + bg
    max_val = np.max(np.abs(mixed)) + 1e-8
    if max_val > 1.0:
        mixed = mixed / max_val
    return mixed.astype(np.float32)


def _rawboost_1d(xi: np.ndarray, sr: int) -> np.ndarray:
    """Apply LnL convolutive noise + optional SSI and ISD additive noise."""
    y = LnL_convolutive_noise(
        xi, N_f=5, nBands=5,
        minF=20.0, maxF=8000.0,
        minBW=100.0, maxBW=1000.0,
        minCoeff=10, maxCoeff=100,
        minG=0.0, maxG=0.0,
        minBiasLinNonLin=5.0, maxBiasLinNonLin=20.0,
        fs=sr,
    )
    if random.random() < 0.5:
        y = SSI_additive_noise(y, SNRmin=10.0, SNRmax=40.0, nBands=5,
                               minF=20.0, maxF=8000.0, minBW=100.0, maxBW=1000.0,
                               minCoeff=10, maxCoeff=100, minG=0.0, maxG=0.0, fs=sr)
    if random.random() < 0.5:
        y = ISD_additive_noise(y, P=10.0, g_sd=2.0)
    return y


def _aug_one(xi: np.ndarray, sr: int, bg_files: list) -> np.ndarray:
    """Apply the 4-way split augmentation to a single 1-D waveform."""
    p = random.random()
    target_len = xi.shape[0]
    if p < 0.20:
        xi = _rawboost_1d(xi, sr)
    elif p < 0.40:
        xi = mix_mild_background(xi, sr, bg_files)
    elif p < 0.60:
        xi = mix_mild_background(xi, sr, bg_files)
        xi = _rawboost_1d(xi, sr)
    # p >= 0.60: no augmentation
    if xi.shape[0] > target_len:
        xi = xi[:target_len]
    elif xi.shape[0] < target_len:
        xi = np.pad(xi, (0, target_len - xi.shape[0]))
    return xi


# ---------------------------------------------------------------------------
# Batch-level functions used by training pipelines
# ---------------------------------------------------------------------------

def apply_aug_split_batch(x: torch.Tensor, cfg) -> torch.Tensor:
    """4-way split augmentation for stage1 training (reads cfg.bg_files / cfg.target_sample_rate)."""
    device = x.device
    a = x.detach().cpu().numpy()
    for i in range(a.shape[0]):
        xi = a[i].ravel()
        xi = _aug_one(xi, cfg.target_sample_rate, cfg.bg_files)
        a[i] = xi.reshape(a[i].shape)
    return torch.from_numpy(a).to(device=device, dtype=x.dtype)


def apply_aug_split_batch_baseline(x: torch.Tensor, sr: int = 16000,
                                   bg_noise_dir: str = _DEFAULT_BG_NOISE_DIR) -> torch.Tensor:
    """4-way split augmentation for baseline_train (discovers bg files from bg_noise_dir)."""
    bg_files = _discover_bg_files(bg_noise_dir)
    device = x.device
    a = x.detach().cpu().numpy()
    for i in range(a.shape[0]):
        xi = a[i].ravel()
        xi = _aug_one(xi, sr, bg_files)
        a[i] = xi.reshape(a[i].shape)
    return torch.from_numpy(a).to(device=device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Dataset_safe_train_BG_split (original, copied verbatim for reference)
# Source: /data/FF_V2/Speaker_Specific_OCSVM/Mixture/data_utils.py
# ---------------------------------------------------------------------------

try:
    from torch import Tensor
    from torch.utils.data import Dataset
    from RawBoost import process_Rawboost_feature
    _DATASET_CLASS_AVAILABLE = True
except ImportError:
    _DATASET_CLASS_AVAILABLE = False

try:
    from utils import pad as _pad_fn
except ImportError:
    def _pad_fn(x, cut):
        if len(x) >= cut:
            return x[:cut]
        return np.pad(x, (0, cut - len(x)))


class Dataset_safe_train_BG_split(Dataset):
    def __init__(self, args, list_IDs, labels, scores, base_dir, algo, bg_dir=None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.scores = scores
        self.base_dir = base_dir
        self.algo = algo
        self.args = args

        self.cut = 66800

        if bg_dir is not None:
            self.bg_dir = bg_dir
        else:
            self.bg_dir = _DEFAULT_BG_NOISE_DIR

        self.bg_files = glob.glob(os.path.join(self.bg_dir, "*.mp3"))

        self.bg_snr_range = (18, 28)

    def __len__(self):
        return len(self.list_IDs)

    def mix_mild_background(self, speech, sr):
        if len(self.bg_files) == 0:
            return speech
        bg_path = random.choice(self.bg_files)
        bg, _ = librosa.load(bg_path, sr=sr, mono=True)
        if len(bg) == 0:
            return speech
        if len(bg) < len(speech):
            repeat_count = int(np.ceil(len(speech) / len(bg)))
            bg = np.tile(bg, repeat_count)
        start = random.randint(0, len(bg) - len(speech))
        bg = bg[start:start + len(speech)]
        bg = bg - np.mean(bg)
        speech_power = np.mean(speech ** 2) + 1e-8
        bg_power = np.mean(bg ** 2) + 1e-8
        snr_db = random.uniform(self.bg_snr_range[0], self.bg_snr_range[1])
        target_bg_power = speech_power / (10 ** (snr_db / 10))
        bg = bg * np.sqrt(target_bg_power / bg_power)
        mixed = speech + bg
        max_val = np.max(np.abs(mixed)) + 1e-8
        if max_val > 1.0:
            mixed = mixed / max_val
        return mixed.astype(np.float32)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]

        audio_path = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(audio_path, sr=16000, mono=True)

        scores = self.scores[utt_id]

        Y = X

        p = random.random()

        if p < 0.20:
            Y = process_Rawboost_feature(Y, fs, self.args, self.algo)
            scores = scores - 1

        elif p < 0.40:
            Y = self.mix_mild_background(Y, fs)
            Y = process_Rawboost_feature(Y, fs, self.args, 0)
            scores = scores - 0.5

        elif p < 0.60:
            Y = self.mix_mild_background(Y, fs)
            Y = process_Rawboost_feature(Y, fs, self.args, self.algo)
            scores = scores - 1.5

        else:
            Y = process_Rawboost_feature(Y, fs, self.args, 0)

        scores = max(scores, 0)

        X_pad = _pad_fn(Y, self.cut)
        x_inp = Tensor(X_pad)

        target = self.labels[utt_id]

        return x_inp, target, scores, utt_id
