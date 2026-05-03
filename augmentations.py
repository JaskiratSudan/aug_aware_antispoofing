"""
augmentations.py — Audio augmentation for aug_aware_antispoofing.

Augmentation safety tiers:
  Category 1 (always safe as contrastive positives):
    GainVariation, SpeedPerturbation, AddBackgroundNoise, AddReverb
  Category 2 (conditionally safe, constrained severity):
    CodecCompression, BandpassFilter, ModerateResample
  Category 3 (STAGE_2_ONLY — destroys forensic band, forbidden as positives):
    ExtremeResample, LowBitrateCodec
"""

import glob
import os
import random
import subprocess
import tempfile

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF

try:
    from torchaudio.io import AudioEffector, CodecConfig
    _HAS_AUDIO_EFFECTOR = True
except ImportError:
    _HAS_AUDIO_EFFECTOR = False

_SAMPLE_RATE = 16000
_AUDIO_EXTS = {".mp3", ".m4a", ".mov", ".wav", ".flac"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _squeeze(w: torch.Tensor):
    """Return (1d_tensor, original_shape) where 1d_tensor has shape (T,)."""
    orig = w.shape
    if w.ndim == 2 and w.shape[0] == 1:
        return w.squeeze(0), orig
    if w.ndim == 1:
        return w, orig
    raise ValueError(f"Expected 1-D or (1,T) waveform, got shape {w.shape}")


def _restore(w: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
    if len(orig_shape) == 2:
        return w.unsqueeze(0)
    return w


def _fit_length(w: torch.Tensor, T: int) -> torch.Tensor:
    """Trim from end or zero-pad at end to length T."""
    if w.shape[-1] > T:
        return w[..., :T]
    if w.shape[-1] < T:
        pad = T - w.shape[-1]
        return F.pad(w, (0, pad))
    return w


# ---------------------------------------------------------------------------
# Category 1 — Always safe
# ---------------------------------------------------------------------------

class GainVariation:
    """Multiply amplitude by 10^U(-12dB, +6dB) in log space."""

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        w, orig = _squeeze(waveform)
        gain_db = random.uniform(-12.0, 6.0)
        w = w * (10.0 ** (gain_db / 20.0))
        w = w.clamp(-1.0, 1.0)
        return _restore(w, orig)


class SpeedPerturbation:
    """
    Simulate speed factor in [0.93, 1.07] by resampling without pitch correction.
    Output is trimmed from end (speed > 1) or zero-padded at end (speed < 1)
    to preserve original length T.
    """

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        w, orig = _squeeze(waveform)
        T = w.shape[-1]
        speed = random.uniform(0.93, 1.07)
        orig_freq = round(_SAMPLE_RATE * speed)
        if orig_freq == _SAMPLE_RATE:
            return waveform
        w = TAF.resample(w, orig_freq=orig_freq, new_freq=_SAMPLE_RATE)
        w = _fit_length(w, T)
        return _restore(w, orig)


class AddBackgroundNoise:
    """
    Mix speech with a randomly selected background noise at SNR ∈ [snr_min, snr_max] dB.
    Noise files are pre-loaded into memory at __init__ time.
    Supports subdirectory structure (e.g. appliance/, crowd/, guitar/, room/, traffic/, violin/).
    """

    def __init__(self, noise_dir: str, snr_range: tuple = (10.0, 25.0)):
        self._snr_range = snr_range
        self._buffers: list = []

        if not noise_dir:
            return

        all_files = [
            f for f in glob.glob(os.path.join(noise_dir, "**", "*"), recursive=True)
            if os.path.splitext(f)[1].lower() in _AUDIO_EXTS
        ]

        for path in all_files:
            try:
                arr, _ = librosa.load(path, sr=_SAMPLE_RATE, mono=True)
                if len(arr) > 0:
                    self._buffers.append(arr)
            except Exception:
                pass  # skip files that fail to load (e.g. missing ffmpeg codec)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self._buffers:
            return waveform

        w, orig = _squeeze(waveform)
        speech = w.numpy().astype(np.float32)
        T = len(speech)

        bg = random.choice(self._buffers).copy()
        if len(bg) < T:
            bg = np.tile(bg, int(np.ceil(T / len(bg))))
        start = random.randint(0, len(bg) - T)
        bg = bg[start : start + T]
        bg = bg - np.mean(bg)

        speech_power = np.mean(speech ** 2) + 1e-8
        bg_power = np.mean(bg ** 2) + 1e-8
        snr_db = random.uniform(*self._snr_range)
        target_bg_power = speech_power / (10.0 ** (snr_db / 10.0))
        bg = bg * np.sqrt(target_bg_power / bg_power)

        mixed = speech + bg
        peak = np.abs(mixed).max()
        if peak > 1.0:
            mixed /= peak

        return _restore(torch.from_numpy(mixed), orig)


class AddReverb:
    """
    Convolve speech with a random RIR via FFT convolution.
    RIR files are pre-loaded into memory at __init__ time.
    """

    def __init__(self, rir_dir: str):
        self._rir_tensors: list = []

        if not rir_dir:
            return

        for path in glob.glob(os.path.join(rir_dir, "*.wav")):
            try:
                arr, _ = librosa.load(path, sr=_SAMPLE_RATE, mono=True)
                if len(arr) > 0:
                    self._rir_tensors.append(torch.from_numpy(arr.astype(np.float32)))
            except Exception:
                pass

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self._rir_tensors:
            return waveform

        w, orig = _squeeze(waveform)
        T = w.shape[-1]
        rir = random.choice(self._rir_tensors)
        rir = rir / (rir.abs().max() + 1e-8)

        n = T + rir.shape[-1] - 1
        n_fft = 1 << (n - 1).bit_length()  # next power of 2

        out = torch.fft.irfft(
            torch.fft.rfft(w, n=n_fft) * torch.fft.rfft(rir, n=n_fft),
            n=n_fft,
        )[:T]
        peak = out.abs().max()
        if peak > 1.0:
            out = out / peak

        return _restore(out, orig)


# ---------------------------------------------------------------------------
# Category 2 — Conditionally safe (constrained severity)
# ---------------------------------------------------------------------------

class CodecCompression:
    """
    Encode+decode via a safe codec at moderate bitrate.
    Safe options: mp3@64k, mp3@128k, aac@64k, opus@48k.
    Uses torchaudio AudioEffector when available; falls back to ffmpeg subprocess.
    """

    _FFMPEG_CONFIGS = [
        ("mp3",  "libmp3lame", "64k",  ".mp3"),
        ("mp3",  "libmp3lame", "128k", ".mp3"),
        ("aac",  "aac",        "64k",  ".m4a"),
        ("opus", "libopus",    "48k",  ".ogg"),
    ]

    def __init__(self):
        self._effectors = None
        if _HAS_AUDIO_EFFECTOR:
            try:
                self._effectors = [
                    AudioEffector(format="mp3", encoder="libmp3lame",
                                  codec_config=CodecConfig(bit_rate=64_000)),
                    AudioEffector(format="mp3", encoder="libmp3lame",
                                  codec_config=CodecConfig(bit_rate=128_000)),
                    AudioEffector(format="mp4", encoder="aac",
                                  codec_config=CodecConfig(bit_rate=64_000)),
                    AudioEffector(format="ogg", encoder="libopus",
                                  codec_config=CodecConfig(bit_rate=48_000)),
                ]
            except Exception:
                self._effectors = None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        w, orig = _squeeze(waveform)
        T_orig = w.shape[-1]

        if self._effectors is not None:
            return self._apply_effector(w, orig, T_orig)
        return self._apply_ffmpeg(w, orig, T_orig)

    def _apply_effector(self, w, orig, T_orig):
        try:
            eff = random.choice(self._effectors)
            wav_2d = w.unsqueeze(1).float()          # (T, 1)
            out_2d = eff.apply(wav_2d, sample_rate=_SAMPLE_RATE)  # (T', 1)
            out = out_2d.squeeze(1)                  # (T',)
            out = _fit_length(out, T_orig)
            return _restore(out, orig)
        except Exception:
            return _restore(w, orig)

    def _apply_ffmpeg(self, w, orig, T_orig):
        _, fmt, bitrate, ext = random.choice(self._FFMPEG_CONFIGS)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                in_wav  = os.path.join(tmp, "in.wav")
                out_enc = os.path.join(tmp, f"out{ext}")
                out_wav = os.path.join(tmp, "out.wav")

                # write input wav via torchaudio
                import torchaudio
                torchaudio.save(in_wav, w.unsqueeze(0), _SAMPLE_RATE)

                subprocess.run(
                    ["ffmpeg", "-y", "-i", in_wav,
                     "-codec:a", fmt, "-b:a", bitrate, out_enc],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-i", out_enc, out_wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                arr, _ = librosa.load(out_wav, sr=_SAMPLE_RATE, mono=True)
                out = torch.from_numpy(arr.astype(np.float32))
                out = _fit_length(out, T_orig)
                return _restore(out, orig)
        except Exception:
            return _restore(w, orig)


class BandpassFilter:
    """
    High-pass at random cutoff ∈ [80, 300] Hz + low-pass at random cutoff ∈ [6000, 7500] Hz.
    Never cuts below 6 kHz on the high end.
    """

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        hp_cutoff = random.uniform(80.0, 300.0)
        lp_cutoff = random.uniform(6000.0, 7500.0)
        w = TAF.highpass_biquad(waveform, sample_rate=_SAMPLE_RATE, cutoff_freq=hp_cutoff)
        w = TAF.lowpass_biquad(w,         sample_rate=_SAMPLE_RATE, cutoff_freq=lp_cutoff)
        return w


class ModerateResample:
    """
    Downsample to 22050 or 24000 Hz, then back to 16000 Hz.
    Never goes below 22050 Hz (preserves most of the forensic band above 4 kHz).
    """

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        w, orig = _squeeze(waveform)
        T_orig = w.shape[-1]
        target_sr = random.choice([22050, 24000])
        w = TAF.resample(w, orig_freq=_SAMPLE_RATE, new_freq=target_sr)
        w = TAF.resample(w, orig_freq=target_sr,    new_freq=_SAMPLE_RATE)
        w = _fit_length(w, T_orig)
        return _restore(w, orig)


# ---------------------------------------------------------------------------
# Category 3 — STAGE_2_ONLY (never used as contrastive positives)
# ---------------------------------------------------------------------------

class ExtremeResample:
    """
    CATEGORY 3 — STAGE_2_ONLY.
    Downsamples to 8000 Hz, destroying the 4-8 kHz forensic band used for
    deepfake detection. Do NOT use as a positive view in Stage 1 NT-Xent training.
    """

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        w, orig = _squeeze(waveform)
        T_orig = w.shape[-1]
        w = TAF.resample(w, orig_freq=_SAMPLE_RATE, new_freq=8000)
        w = TAF.resample(w, orig_freq=8000,          new_freq=_SAMPLE_RATE)
        w = _fit_length(w, T_orig)
        return _restore(w, orig)


class LowBitrateCodec:
    """
    CATEGORY 3 — STAGE_2_ONLY.
    Encodes at ≤16 kbps (opus@8k or opus@16k), destroying spectral fine structure
    that TTS artifacts live in. Do NOT use as a positive view in Stage 1.
    """

    _CONFIGS = [
        CodecConfig(bit_rate=8_000) if _HAS_AUDIO_EFFECTOR else None,
        CodecConfig(bit_rate=16_000) if _HAS_AUDIO_EFFECTOR else None,
    ]

    def __init__(self):
        self._effectors = None
        if _HAS_AUDIO_EFFECTOR:
            try:
                self._effectors = [
                    AudioEffector(format="ogg", encoder="libopus",
                                  codec_config=CodecConfig(bit_rate=8_000)),
                    AudioEffector(format="ogg", encoder="libopus",
                                  codec_config=CodecConfig(bit_rate=16_000)),
                ]
            except Exception:
                pass

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self._effectors:
            return waveform
        w, orig = _squeeze(waveform)
        T_orig = w.shape[-1]
        try:
            eff = random.choice(self._effectors)
            wav_2d = w.unsqueeze(1).float()
            out = eff.apply(wav_2d, sample_rate=_SAMPLE_RATE).squeeze(1)
            out = _fit_length(out, T_orig)
            return _restore(out, orig)
        except Exception:
            return _restore(w, orig)


# ---------------------------------------------------------------------------
# AudioAugmentor
# ---------------------------------------------------------------------------

class AudioAugmentor:
    """
    Applies audio augmentation to generate multiple positive views for NT-Xent training.

    n_views: total number of augmented views returned by get_views / get_views_batch.
             All views are independently augmented — no separate "clean anchor" concept.

    mode='single': apply exactly one randomly sampled augmentation from Cat1 + Cat2.
    mode='chain':  apply 1-2 Cat1 augs in sequence, optionally followed by one Cat2 aug.

    If noise_dir or rir_dir is empty/None, the corresponding augmentation is skipped
    and the pool falls back to Cat1-only (GainVariation + SpeedPerturbation always present).
    """

    def __init__(
        self,
        noise_dir: str = None,
        rir_dir: str = None,
        n_views: int = 5,
        mode: str = "single",
    ):
        self.n_views = n_views
        self.mode = mode

        self._cat1 = [GainVariation(), SpeedPerturbation()]
        if noise_dir:
            bg = AddBackgroundNoise(noise_dir)
            if bg._buffers:
                self._cat1.append(bg)
        if rir_dir:
            rv = AddReverb(rir_dir)
            if rv._rir_tensors:
                self._cat1.append(rv)

        # CodecCompression excluded — AudioEffector roundtrip takes 1-3s per clip,
        # making DataLoader workers the bottleneck. BandpassFilter + ModerateResample
        # cover the same perceptual-level degradation without the CPU cost.
        self._cat2 = [BandpassFilter(), ModerateResample()]

    # ------------------------------------------------------------------
    def _apply_single(self, waveform: torch.Tensor) -> torch.Tensor:
        pool = self._cat1 + self._cat2
        aug = random.choice(pool)
        return aug(waveform)

    def _apply_chain(self, waveform: torch.Tensor) -> torch.Tensor:
        n = random.randint(1, 2)
        cat1_augs = random.sample(self._cat1, min(n, len(self._cat1)))
        w = waveform
        for aug in cat1_augs:
            w = aug(w)
        if random.random() < 0.5:
            cat2_aug = random.choice(self._cat2)
            w = cat2_aug(w)
        return w

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.mode == "chain":
            return self._apply_chain(waveform)
        return self._apply_single(waveform)

    def get_views(self, waveform: torch.Tensor, n: int = None) -> list:
        """Return n independently augmented views of waveform."""
        n = n if n is not None else self.n_views
        return [self(waveform) for _ in range(n)]

    def get_views_batch(self, waveforms: torch.Tensor, n: int = None) -> torch.Tensor:
        """
        waveforms: (B, T) CPU tensor
        returns:   (B, n_views, T) CPU tensor
        """
        n = n if n is not None else self.n_views
        B, T = waveforms.shape
        result = torch.zeros(B, n, T, dtype=waveforms.dtype)
        for i in range(B):
            views = self.get_views(waveforms[i], n)  # list of n tensors shape (T,)
            for j, v in enumerate(views):
                result[i, j] = v[:T] if v.shape[-1] >= T else F.pad(v, (0, T - v.shape[-1]))
        return result
