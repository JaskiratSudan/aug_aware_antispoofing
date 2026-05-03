from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import librosa


class BaseAudioDataset(torch.utils.data.Dataset):
    """
    A base class for audio datasets to handle common processing tasks.
    Tracks how many files were successfully decoded vs. failed.
    """
    loaded_count = 0
    failed_count = 0

    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
        repeat_short: bool = True,
        **kwargs,
    ):
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.repeat_short = repeat_short

    def _pad_or_repeat(self, waveform: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = waveform.shape[0]
        if current_len >= target_len:
            return waveform[:target_len]
        if self.repeat_short and current_len > 0:
            reps = (target_len + current_len - 1) // current_len
            return waveform.repeat(reps)[:target_len]
        return F.pad(waveform, (0, target_len - current_len))

    def _process_audio(self, audio_path: Path) -> torch.Tensor:
        try:
            waveform, sample_rate = librosa.load(
                audio_path, sr=self.target_sample_rate, mono=True
            )
            waveform = torch.from_numpy(waveform).float()
            BaseAudioDataset.loaded_count += 1
        except Exception as e:
            tqdm.write(f"[WARNING] Corrupted file: {audio_path}. Error: {e}")
            BaseAudioDataset.failed_count += 1
            if self.max_duration_seconds is not None:
                return torch.zeros(self.max_duration_seconds * self.target_sample_rate)
            else:
                return torch.zeros(self.target_sample_rate)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if self.max_duration_seconds is not None:
            target_len = self.max_duration_seconds * self.target_sample_rate
            current_len = waveform.shape[0]
            if current_len > target_len:
                waveform = waveform[:target_len]

        return waveform

    @classmethod
    def print_summary(cls):
        total = cls.loaded_count + cls.failed_count
        print(f"\n[DATASET SUMMARY] Loaded: {cls.loaded_count}, Failed: {cls.failed_count}, Total: {total}")
