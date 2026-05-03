# asvspoof_windowed_loader.py
# Windowed ASVspoof loaders with tail-drop and padding rules.

from pathlib import Path
import random
import torch
import torch.nn.functional as F
import librosa


def _load_and_chunk(
    audio_path: Path,
    target_sample_rate: int,
    window_seconds: float,
    min_tail_seconds: float,
) -> torch.Tensor:
    target_len = int(window_seconds * target_sample_rate)
    min_tail_len = int(min_tail_seconds * target_sample_rate)
    try:
        waveform, _ = librosa.load(audio_path, sr=target_sample_rate, mono=True)
        waveform = torch.from_numpy(waveform).float()
    except Exception:
        return torch.zeros((1, target_len), dtype=torch.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    length = waveform.shape[0]
    if length <= target_len:
        padded = F.pad(waveform, (0, target_len - length))
        return padded.unsqueeze(0)

    chunks = []
    start = 0
    while start + target_len <= length:
        chunks.append(waveform[start:start + target_len])
        start += target_len

    remainder = length - start
    if remainder >= min_tail_len:
        tail = waveform[start:length]
        tail = F.pad(tail, (0, target_len - remainder))
        chunks.append(tail)

    if not chunks:
        return torch.zeros((1, target_len), dtype=torch.float32)
    return torch.stack(chunks, dim=0)


class ASVspoof2019WindowedDataset(torch.utils.data.Dataset):
    """
    ASVspoof2019 with windowed audio:
      - window_seconds length chunks
      - tail < min_tail_seconds is dropped
      - short audio is zero-padded to window length
    Returns (waveforms, binary_label, multi_label, speaker_id, audio_name)
    """
    def __init__(
        self,
        protocol_file: str,
        root_dir: str,
        subset: str = "all",
        num_samples: int = None,
        sample_seed: int = 1337,
        target_sample_rate: int = 16000,
        window_seconds: float = 5.0,
        min_tail_seconds: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.data = []
        self.attack_to_idx = {"bonafide": 0}
        self.target_sample_rate = target_sample_rate
        self.window_seconds = window_seconds
        self.min_tail_seconds = min_tail_seconds

        subset = (subset or "all").lower()
        if subset not in {"all", "bonafide", "spoof"}:
            raise ValueError(
                f"subset must be one of 'all', 'bonafide', or 'spoof' (got: {subset})"
            )

        sample_limit = int(num_samples) if num_samples is not None else None
        seen = 0
        rng = random.Random(sample_seed)

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                audio_rel = parts[0]
                attack_id_raw = parts[1]
                label_str = parts[2].lower()
                speaker_id = parts[4]

                if subset != "all" and label_str != subset:
                    continue

                audio_name = audio_rel.split("/").pop()
                full_path = self.root_dir / audio_name
                binary_label = 1 if label_str == "bonafide" else 0

                if label_str == "bonafide":
                    key = "bonafide"
                else:
                    key = attack_id_raw
                if key not in self.attack_to_idx:
                    self.attack_to_idx[key] = len(self.attack_to_idx)
                multi_label = self.attack_to_idx[key]

                row = (full_path, binary_label, multi_label, speaker_id, audio_name)
                if sample_limit is None:
                    self.data.append(row)
                else:
                    seen += 1
                    if len(self.data) < sample_limit:
                        self.data.append(row)
                    else:
                        j = rng.randint(0, seen - 1)
                        if j < sample_limit:
                            self.data[j] = row

        if not self.data:
            raise RuntimeError(
                f"No audio files found from protocol {protocol_file} "
                f"after applying subset='{subset}'."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, binary_label, multi_label, speaker_id, audio_name = self.data[idx]
        waveforms = _load_and_chunk(
            audio_path,
            target_sample_rate=self.target_sample_rate,
            window_seconds=self.window_seconds,
            min_tail_seconds=self.min_tail_seconds,
        )
        return (
            waveforms,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.long),
            speaker_id,
            audio_name,
        )


class ASVspoof5WindowedDataset(torch.utils.data.Dataset):
    """
    ASVspoof5 with windowed audio:
      - window_seconds length chunks
      - tail < min_tail_seconds is dropped
      - short audio is zero-padded to window length
    Returns (waveforms, binary_label, multi_label, speaker_id, audio_name)
    """
    def __init__(
        self,
        protocol_file: str,
        root_dir: str = "",
        num_samples: int = None,
        subset: str = "all",
        sample_seed: int = 1337,
        file_ext: str = ".flac",
        target_sample_rate: int = 16000,
        window_seconds: float = 5.0,
        min_tail_seconds: float = 1.0,
    ):
        self.root_dir = Path(root_dir) if root_dir else None
        self.data = []
        self.attack_to_idx = {"bonafide": 0}
        self.target_sample_rate = target_sample_rate
        self.window_seconds = window_seconds
        self.min_tail_seconds = min_tail_seconds

        subset = (subset or "all").lower()
        if subset not in {"all", "bonafide", "spoof"}:
            raise ValueError(
                f"subset must be one of 'all', 'bonafide', or 'spoof' (got: {subset})"
            )

        def _with_ext(name: str) -> str:
            if Path(name).suffix:
                return name
            return f"{name}{file_ext}"

        sample_limit = int(num_samples) if num_samples is not None else None
        seen = 0
        rng = random.Random(sample_seed)

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                speaker_id = parts[0]
                file_name = parts[1]
                attack_id_raw = parts[4]
                label_str = parts[5].lower()

                if subset != "all" and label_str != subset:
                    continue

                audio_name = _with_ext(file_name)
                if self.root_dir is not None:
                    full_path = self.root_dir / audio_name
                else:
                    full_path = Path(audio_name)

                binary_label = 1 if label_str == "bonafide" else 0
                key = "bonafide" if label_str == "bonafide" else attack_id_raw
                if key not in self.attack_to_idx:
                    self.attack_to_idx[key] = len(self.attack_to_idx)
                multi_label = self.attack_to_idx[key]

                row = (full_path, binary_label, multi_label, speaker_id, audio_name)
                if sample_limit is None:
                    self.data.append(row)
                else:
                    seen += 1
                    if len(self.data) < sample_limit:
                        self.data.append(row)
                    else:
                        j = rng.randint(0, seen - 1)
                        if j < sample_limit:
                            self.data[j] = row

        if not self.data:
            raise RuntimeError(
                f"No audio files found from protocol {protocol_file} "
                f"after applying subset='{subset}'."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, binary_label, multi_label, speaker_id, audio_name = self.data[idx]
        waveforms = _load_and_chunk(
            audio_path,
            target_sample_rate=self.target_sample_rate,
            window_seconds=self.window_seconds,
            min_tail_seconds=self.min_tail_seconds,
        )
        return (
            waveforms,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.long),
            speaker_id,
            audio_name,
        )


class MLAADMailabsWindowedDataset(torch.utils.data.Dataset):
    """
    MLAAD/MAILabs with windowed audio.
    Protocol format:
      rel_path lang corpus source_or_attack label
    Returns (waveforms, binary_label, multi_label, audio_name)
    """
    def __init__(
        self,
        protocol_file: str,
        root_dir: str,
        num_samples: int = None,
        subset: str = "all",
        sample_seed: int = 1337,
        target_sample_rate: int = 16000,
        window_seconds: float = 5.0,
        min_tail_seconds: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.data = []
        self.attack_to_idx = {"bonafide": 0}
        self.target_sample_rate = target_sample_rate
        self.window_seconds = window_seconds
        self.min_tail_seconds = min_tail_seconds

        subset = (subset or "all").lower()
        if subset not in {"all", "bonafide", "spoof"}:
            raise ValueError(
                f"subset must be one of 'all', 'bonafide', or 'spoof' (got: {subset})"
            )

        sample_limit = int(num_samples) if num_samples is not None else None
        seen = 0
        rng = random.Random(sample_seed)

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                rel_path = parts[0]
                source_or_attack = parts[3]
                label_str = parts[4].lower()

                if subset != "all" and label_str != subset:
                    continue

                audio_name = Path(rel_path).name
                full_path = self.root_dir / rel_path
                binary_label = 1 if label_str == "bonafide" else 0

                key = "bonafide" if label_str == "bonafide" else source_or_attack
                if key not in self.attack_to_idx:
                    self.attack_to_idx[key] = len(self.attack_to_idx)
                multi_label = self.attack_to_idx[key]

                row = (full_path, binary_label, multi_label, audio_name)
                if sample_limit is None:
                    self.data.append(row)
                else:
                    seen += 1
                    if len(self.data) < sample_limit:
                        self.data.append(row)
                    else:
                        j = rng.randint(0, seen - 1)
                        if j < sample_limit:
                            self.data[j] = row

        if not self.data:
            raise RuntimeError(
                f"No audio files found from protocol {protocol_file} "
                f"after applying subset='{subset}'."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, binary_label, multi_label, audio_name = self.data[idx]
        waveforms = _load_and_chunk(
            audio_path,
            target_sample_rate=self.target_sample_rate,
            window_seconds=self.window_seconds,
            min_tail_seconds=self.min_tail_seconds,
        )
        return (
            waveforms,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.long),
            audio_name,
        )
