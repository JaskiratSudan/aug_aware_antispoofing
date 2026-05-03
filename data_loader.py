# data_loader.py
# This version is updated to include a wrapper for applying augmentations.

import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
import soundfile
import librosa
import random

from collate import (
    pad_collate_fn,
    pad_collate_fn_aug,
    pad_collate_fn_speaker,
    pad_collate_fn_speaker_source,
    pad_collate_fn_speaker_source_multiclass,
)
from base_audio import BaseAudioDataset

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")

def load_fakexpose_items(root_dir: str, allowed_exts=None):
    if allowed_exts is None:
        allowed_exts = {".wav", ".flac", ".mp3", ".m4a"}

    root = Path(root_dir)
    fake_candidates = ["ElevenLabs", "11 labs", "11_labs", "11labs"]
    real_candidates = ["Original", "original"]

    fake_dir = None
    real_dir = None

    for name in fake_candidates:
        cand = root / name
        if cand.exists():
            fake_dir = cand
            break
    for name in real_candidates:
        cand = root / name
        if cand.exists():
            real_dir = cand
            break

    missing = []
    if fake_dir is None:
        missing.append(f"{root}/<{','.join(fake_candidates)}>")
    if real_dir is None:
        missing.append(f"{root}/<{','.join(real_candidates)}>")
    if missing:
        raise FileNotFoundError(f"Missing Fakexpose directories: {missing}")

    items = []

    def _collect(dir_path: Path, label: int, source: str):
        for path in dir_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in allowed_exts:
                items.append((path, label, source))

    _collect(fake_dir, 0, "elevenlabs")
    _collect(real_dir, 1, "original")

    items.sort(key=lambda x: str(x[0]))
    return items

# ---------- New Dataset: FamousFigures ----------
class FamousFiguresDataset(BaseAudioDataset):
    """
    FamousFigures dataset using a protocol file with columns:
      AudioName, Speaker, Source, Label, AudioPath

    Returns: (waveform, label_tensor, speaker_str, source_str)
      - label: 1 for bonafide, 0 for spoof (case-insensitive; also normalizes 'bona-fide')
    """
    def __init__(
        self,
        protocol_file: str,
        root_dir: str = "",
        subset: str = "all",             # 'all' | 'bonafide' | 'spoof'
        include_speakers: list = None,   # optional allowlist of speakers
        include_sources: list = None,    # optional allowlist of sources
        return_audio_name: bool = False, # return audio file name for utt_id if True
        **kwargs
    ):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir) if root_dir else None
        self.return_audio_name = return_audio_name

        # Load protocol; tabs or general whitespace both OK
        try:
            df = pd.read_csv(protocol_file, sep="\t")
        except Exception:
            df = pd.read_csv(protocol_file, sep=r"\s+", engine="python")

        # Normalize expected columns
        expected = {"AudioName", "Speaker", "Source", "Label", "AudioPath"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Protocol is missing columns: {sorted(missing)}")

        # Normalize labels
        df["Label"] = df["Label"].astype(str).str.lower().str.replace("bona-fide", "bonafide")

        # Clean obviously malformed paths by trimming after first '.wav'
        def _clean_path(p):
            s = str(p)
            if ".wav" in s:
                s = s[: s.lower().find(".wav") + 4]
            return s

        df["AudioPath"] = df["AudioPath"].astype(str).map(_clean_path)

        # Make paths absolute if root_dir provided and path is relative
        if self.root_dir is not None:
            df["AudioPath"] = df["AudioPath"].apply(
                lambda p: str(Path(p)) if Path(p).is_absolute() else str(self.root_dir / p)
            )

        # Subset by label
        if subset == "bonafide":
            df = df[df["Label"] == "bonafide"]
        elif subset == "spoof":
            df = df[df["Label"] != "bonafide"]

        # Optional filters
        if include_speakers:
            keep = set(map(str, include_speakers))
            df = df[df["Speaker"].astype(str).isin(keep)]
        if include_sources:
            keep = set(map(str, include_sources))
            df = df[df["Source"].astype(str).isin(keep)]

        # Keep only rows whose files exist
        df["exists"] = df["AudioPath"].apply(lambda p: Path(p).exists())
        missing_n = int((~df["exists"]).sum())
        if missing_n > 0:
            print(f"[INFO] FamousFigures: filtered out {missing_n} missing audio files.")
        df = df[df["exists"]].copy()

        # Optional sampling (shuffle for variety)
        if num_samples is not None and len(df) > num_samples:
            df = df.sample(frac=1, random_state=42).head(num_samples)

        if len(df) == 0:
            raise RuntimeError("FamousFiguresDataset: No audio after filtering.")

        # Store compact rows for fast indexing
        # Map label to 1/0 (bonafide=1, spoof/other=0)
        def _lbl_to_int(lbl: str) -> int:
            return 1 if str(lbl).lower() == "bonafide" else 0

        self.rows = [
            (Path(row["AudioPath"]), _lbl_to_int(row["Label"]), str(row["Speaker"]), str(row["Source"]))
            for _, row in df.iterrows()
        ]

        print(f"[INFO] FamousFigures: loaded {len(self.rows)} samples (subset={subset}).")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        audio_path, label_int, speaker, source = self.rows[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(label_int, dtype=torch.long)
        if self.return_audio_name:
            audio_name = Path(audio_path).name
            return waveform, label, speaker, source, audio_name
        return waveform, label, speaker, source

class ASVspoof2019Dataset(BaseAudioDataset):
    def __init__(
        self,
        protocol_file: str,
        root_dir: str = "",
        num_samples: int = None,
        subset: str = "all",
        sample_seed: int = 1337,   # <— NEW (optional)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.data = []
        self.attack_to_idx = {"bonafide": 0}  # multi-class: bonafide -> 0

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
                attack_id_raw = parts[1]          # e.g., A11 or '-' for bonafide
                label_str = parts[2].lower()      # 'bonafide' or 'spoof'
                speaker_id = parts[4]             # e.g., p240

                # subset filter
                if subset != "all" and label_str != subset:
                    continue

                # file name only
                audio_name = audio_rel.split("/").pop()
                full_path = self.root_dir / audio_name

                # Binary label for convenience
                binary_label = 1 if label_str == "bonafide" else 0

                # Multi-class label:
                if label_str == "bonafide":
                    key = "bonafide"
                else:
                    # spoof -> use attack ID (e.g., A11)
                    key = attack_id_raw

                if key not in self.attack_to_idx:
                    self.attack_to_idx[key] = len(self.attack_to_idx)

                multi_label = self.attack_to_idx[key]

                # Store both labels
                self.data.append((full_path, binary_label, multi_label, speaker_id, audio_name))

        if num_samples is not None:
            n = min(int(num_samples), len(self.data))
            rng = random.Random(sample_seed)
            self.data = rng.sample(self.data, n)

        if not self.data:
            raise RuntimeError(
                f"No audio files found from protocol {protocol_file} "
                f"after applying subset='{subset}'."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, binary_label, multi_label, speaker_id, audio_name = self.data[idx]
        waveform = self._process_audio(audio_path)
        return (
            waveform,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.long),
            speaker_id,
            audio_name,
        )

class ASVspoof5Dataset(BaseAudioDataset):
    """
    ASVspoof5 protocol format (whitespace-separated):
      SPEAKER_ID | FILE | GENDER | CODEC | ATTACK | LABEL
    Returns (waveform, binary_label, multi_label, speaker_id, audio_name)
    """
    def __init__(
        self,
        protocol_file: str,
        root_dir: str = "",
        num_samples: int = None,
        subset: str = "all",
        sample_seed: int = 1337,
        file_ext: str = ".flac",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir) if root_dir else None
        self.data = []
        self.attack_to_idx = {"bonafide": 0}

        subset = (subset or "all").lower()
        if subset not in {"all", "bonafide", "spoof"}:
            raise ValueError(
                f"subset must be one of 'all', 'bonafide', or 'spoof' (got: {subset})"
            )

        def _with_ext(name: str) -> str:
            return name if Path(name).suffix else f"{name}{file_ext}"

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
                full_path = (self.root_dir / audio_name) if self.root_dir else Path(audio_name)
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
        waveform = self._process_audio(audio_path)
        return (
            waveform,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.long),
            speaker_id,
            audio_name,
        )

class MLAADMailabsDataset(BaseAudioDataset):
    """
    Protocol format (whitespace-separated):
      rel_path lang corpus source_or_attack label
    Example:
      MLAAD/fake/it/...wav it MLAAD tts_models_it_mai_male_vits spoof
      MAILabs/...wav de MAILABS - bonafide
    Returns (waveform, binary_label, multi_label, audio_name)
    """
    def __init__(
        self,
        protocol_file: str,
        root_dir: str,
        num_samples: int = None,
        subset: str = "all",
        sample_seed: int = 1337,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.data = []
        self.attack_to_idx = {"bonafide": 0}

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
        waveform = self._process_audio(audio_path)
        if self.max_duration_seconds is not None:
            target_len = int(self.max_duration_seconds * self.target_sample_rate)
            if waveform.shape[0] < target_len:
                waveform = F.pad(waveform, (0, target_len - waveform.shape[0]))
        return (
            waveform,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.long),
            audio_name,
        )

class RAVDESSDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the RAVDESS dataset."""
    def __init__(self, root_dir: str, **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)
        
        self.root_dir = Path(root_dir)
        self.audio_files = sorted(list(self.root_dir.glob('**/Actor_*/*.wav')))
        
        if num_samples is not None:
            self.audio_files = self.audio_files[:num_samples]
            
        if not self.audio_files:
            raise RuntimeError(f"No .wav files found in {root_dir}.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1, dtype=torch.long)
        return waveform, label

class CommonVoiceDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the Common Voice dataset."""
    def __init__(self, root_dir: str, **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.audio_files = sorted(list(self.root_dir.glob('**/*.wav')))
        
        if num_samples is not None:
            self.audio_files = self.audio_files[:num_samples]
            
        if not self.audio_files:
            raise RuntimeError(f"No .wav files found in {root_dir}.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1, dtype=torch.long)
        return waveform, label

class ASVspoof2021Dataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the ASVspoof 2021 DF eval dataset using ok_files.txt."""
    def __init__(self, root_dir: str, ok_files: str, protocol_file: str, subset="all", **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)
        
        self.root_dir = Path(root_dir)
        self.audio_folder = self.root_dir / "flac"

        # Load ok_files list
        with open(ok_files, "r") as f:
            ok_list = [line.strip() for line in f if line.strip()]
        ok_set = set([Path(x).stem for x in ok_list])  # keep just the stems like DF_E_2000011

        # Load protocol file for labels
        col_names = [
            "speaker_id", "filename", "compression", "source", "attack_id",
            "label", "trim", "set", "vocoder_type", "col10", "col11", "col12", "col13"
        ]
        protocol_df = pd.read_csv(protocol_file, sep="\s+", header=None, engine="python", names=col_names)

        # Filter by ok_files
        protocol_df = protocol_df[protocol_df["filename"].isin(ok_set)]

        if subset == "bonafide":
            self.protocol = protocol_df[protocol_df["label"] == "bonafide"].reset_index(drop=True)
        elif subset == "spoof":
            self.protocol = protocol_df[protocol_df["label"] != "bonafide"].reset_index(drop=True)
        else:
            self.protocol = protocol_df.reset_index(drop=True)

        if num_samples is not None:
            self.protocol = self.protocol.sample(frac=1, random_state=42).reset_index(drop=True).head(num_samples)

        if len(self.protocol) == 0:
            raise RuntimeError(f"Found 0 audio files after filtering with ok_files and subset='{subset}'.")

        print(f"[INFO] Loaded {len(self.protocol)} samples (subset={subset}).")

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]
        audio_path = self.audio_folder / f"{row['filename']}.flac"
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1 if row["label"] == "bonafide" else 0, dtype=torch.long)
        return waveform, label

class InTheWildDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from an In-the-Wild dataset."""
    def __init__(self, root_dir: str, protocol_file: str, subset='all', **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.audio_folder = self.root_dir  # audio files directly here (or adjust if needed)

        if not Path(protocol_file).exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

        # ITW meta.csv has no header row: columns are file, speaker, label
        protocol_df = pd.read_csv(protocol_file, header=None, names=['file', 'speaker', 'label'])
        protocol_df.columns = protocol_df.columns.str.lower()
        if 'audio' in protocol_df.columns and 'file' not in protocol_df.columns:
            protocol_df = protocol_df.rename(columns={'audio': 'file'})

        # Standardize labels: 'bona-fide' -> 'bonafide'
        protocol_df['label'] = protocol_df['label'].replace('bona-fide', 'bonafide')

        # ---------- NEW: detect speaker column ----------
        self.spk_col = "speaker"
        # ------------------------------------------------

        # Keep only rows where audio file exists
        original_count = len(protocol_df)
        protocol_df['exists'] = protocol_df['file'].apply(
            lambda fname: (self.audio_folder / fname).exists()
        )
        protocol_df = protocol_df[protocol_df['exists']]
        if len(protocol_df) < original_count:
            print(f"[INFO] InTheWild: Filtered out {original_count - len(protocol_df)} missing audio files.")

        # Subset filtering
        if subset == 'bonafide':
            self.protocol = protocol_df[protocol_df['label'] == 'bonafide'].reset_index(drop=True)
        elif subset == 'spoof':
            self.protocol = protocol_df[protocol_df['label'] == 'spoof'].reset_index(drop=True)
        else:
            self.protocol = protocol_df

        # Optional sampling
        if num_samples is not None:
            self.protocol = (
                self.protocol
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
                .head(num_samples)
            )

        if len(self.protocol) == 0:
            raise RuntimeError(f"Found 0 audio files after filtering for subset '{subset}'.")

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]

        # file path + audio name
        audio_rel = str(row['file'])
        audio_path = self.audio_folder / audio_rel
        audio_name = Path(audio_rel).name

        # speaker (if column exists; else 'unknown')
        if self.spk_col is not None:
            speaker = str(row[self.spk_col])
        else:
            speaker = "unknown"

        waveform = self._process_audio(audio_path)
        label = torch.tensor(
            1 if row['label'] == 'bonafide' else 0,
            dtype=torch.long
        )

        # ---------- KEY CHANGE: now return 4 items ----------
        return waveform, label, speaker, audio_name

class FakeXposeDataset(BaseAudioDataset):
    """Dataset for Fakexpose: 11 labs (fake) vs original (real)."""
    def __init__(
        self,
        root_dir: str,
        subset: str = "all",
        num_samples: int = None,
        sample_seed: int = 1337,
        allowed_exts=None,
        return_audio_name: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        subset = (subset or "all").lower()
        if subset not in {"all", "bonafide", "spoof"}:
            raise ValueError(
                f"subset must be one of 'all', 'bonafide', or 'spoof' (got: {subset})"
            )

        items = load_fakexpose_items(root_dir, allowed_exts=allowed_exts)

        if subset == "bonafide":
            items = [it for it in items if it[1] == 1]
        elif subset == "spoof":
            items = [it for it in items if it[1] == 0]

        if num_samples is not None:
            n = min(int(num_samples), len(items))
            rng = random.Random(sample_seed)
            items = rng.sample(items, n)

        if not items:
            raise RuntimeError("FakeXposeDataset: no audio files after filtering.")

        self.items = items
        self.return_audio_name = return_audio_name

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, label_int, source = self.items[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(label_int, dtype=torch.long)
        if self.return_audio_name:
            return waveform, label, source, Path(audio_path).name
        return waveform, label, source

# Example of how to use the new InTheWildDataset
if __name__ == '__main__':

    # Path to directory with .flac files
    # ASVSPOOF_TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
    # ASVSPOOF_TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"

    ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
    ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

    # train_dataset = ASVspoof2019Dataset(
    #     root_dir=ASVSPOOF_TRAIN_ROOT,
    #     protocol_file=ASVSPOOF_TRAIN_PROTOCOL,
    #     subset=None,
    #     num_samples=22000,
    #     max_duration_seconds=5
    # )

    train_dataset = InTheWildDataset(
        root_dir=ITW_ROOT,
        protocol_file=ITW_PROTOCOL,
        subset=None,
        num_samples=30,
        max_duration_seconds=5
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=4, collate_fn=pad_collate_fn_speaker_source, pin_memory=True
    )

    for wav, lab, speaker, source in train_loader:
        # print(f"Waveforms: {wav}")
        print(f"Labels: {lab}")
        # print(f"Attack: {attack}")
        print(f"Speaker: {speaker}")


# ---------------------------------------------------------------------------
# Helper + classes needed by eval_datasets.py (added back after revert)
# ---------------------------------------------------------------------------

def _load_asv2021_trials(
    protocol_file: str,
    root_dir: str,
    subset: str,
    num_samples: int,
    sample_seed: int,
    file_ext: str,
    skip_missing: bool,
):
    subset = (subset or "all").lower()
    if subset not in {"all", "bonafide", "spoof"}:
        raise ValueError(
            f"subset must be one of 'all', 'bonafide', or 'spoof' (got: {subset})"
        )

    root = Path(root_dir)
    existing = None
    if skip_missing:
        try:
            existing = {p.stem for p in root.iterdir() if p.is_file() and p.suffix == file_ext}
        except FileNotFoundError:
            existing = set()

    def _with_ext(name: str) -> str:
        return name if Path(name).suffix else f"{name}{file_ext}"

    data = []
    attack_to_idx = {"bonafide": 0}
    missing = 0

    sample_limit = int(num_samples) if num_samples is not None else None
    seen = 0
    rng = random.Random(sample_seed)

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            source_id = parts[0]
            file_id = parts[1]
            attack_id_raw = parts[4]
            label_str = parts[5].lower().replace("bona-fide", "bonafide")

            if subset != "all" and label_str != subset:
                continue

            audio_name = _with_ext(file_id)
            if skip_missing and existing is not None:
                stem = Path(audio_name).stem
                if stem not in existing:
                    missing += 1
                    continue
            full_path = root / audio_name
            if skip_missing and existing is None and not full_path.exists():
                missing += 1
                continue

            binary_label = 1 if label_str == "bonafide" else 0
            key = "bonafide" if label_str == "bonafide" else attack_id_raw
            if key not in attack_to_idx:
                attack_to_idx[key] = len(attack_to_idx)
            multi_label = attack_to_idx[key]

            row = (full_path, binary_label, multi_label, source_id, audio_name)
            if sample_limit is None:
                data.append(row)
            else:
                seen += 1
                if len(data) < sample_limit:
                    data.append(row)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < sample_limit:
                        data[j] = row

    return data, attack_to_idx, missing


class ASVspoof2021DFDataset(BaseAudioDataset):
    """ASVspoof2021 DF eval. Returns (waveform, binary_label, multi_label, speaker_id, audio_name)."""
    def __init__(self, protocol_file, root_dir, num_samples=None, subset="all",
                 sample_seed=1337, file_ext=".flac", skip_missing=True, **kwargs):
        super().__init__(**kwargs)
        self.data, self.attack_to_idx, missing = _load_asv2021_trials(
            protocol_file=protocol_file, root_dir=root_dir, subset=subset,
            num_samples=num_samples, sample_seed=sample_seed,
            file_ext=file_ext, skip_missing=skip_missing,
        )
        if skip_missing and missing > 0:
            print(f"[INFO] ASVspoof2021 DF: skipped {missing} missing files.")
        if not self.data:
            raise RuntimeError(f"No audio files found from protocol {protocol_file}.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        audio_path, binary_label, multi_label, speaker_id, audio_name = self.data[idx]
        waveform = self._process_audio(audio_path)
        return (waveform, torch.tensor(binary_label, dtype=torch.long),
                torch.tensor(multi_label, dtype=torch.long), speaker_id, audio_name)


class ASVspoof2021LADataset(BaseAudioDataset):
    """ASVspoof2021 LA eval. Returns (waveform, binary_label, multi_label, speaker_id, audio_name)."""
    def __init__(self, protocol_file, root_dir, num_samples=None, subset="all",
                 sample_seed=1337, file_ext=".flac", skip_missing=True, **kwargs):
        super().__init__(**kwargs)
        self.data, self.attack_to_idx, missing = _load_asv2021_trials(
            protocol_file=protocol_file, root_dir=root_dir, subset=subset,
            num_samples=num_samples, sample_seed=sample_seed,
            file_ext=file_ext, skip_missing=skip_missing,
        )
        if skip_missing and missing > 0:
            print(f"[INFO] ASVspoof2021 LA: skipped {missing} missing files.")
        if not self.data:
            raise RuntimeError(f"No audio files found from protocol {protocol_file}.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        audio_path, binary_label, multi_label, speaker_id, audio_name = self.data[idx]
        waveform = self._process_audio(audio_path)
        return (waveform, torch.tensor(binary_label, dtype=torch.long),
                torch.tensor(multi_label, dtype=torch.long), speaker_id, audio_name)


class DeepfakeEval2024Dataset(BaseAudioDataset):
    """Deepfake_Eval_2024. Returns (waveform, label_tensor, source_str, audio_name)."""
    def __init__(self, protocol_file, root_dir, subset="all", num_samples=None,
                 sample_seed=1337, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        df = pd.read_csv(protocol_file)
        df.columns = [c.strip() for c in df.columns]
        label_col = df.columns[2]
        df[label_col] = df[label_col].astype(str).str.lower()
        df["full_path"] = df[df.columns[0]].apply(lambda f: self.root_dir / f)
        df["exists"] = df["full_path"].apply(lambda p: Path(p).exists())
        missing = int((~df["exists"]).sum())
        if missing > 0:
            print(f"[INFO] DeepfakeEval2024: filtered out {missing} missing audio files.")
        df = df[df["exists"]].copy()
        if num_samples is not None and len(df) > num_samples:
            df = df.sample(frac=1, random_state=sample_seed).head(num_samples)
        if len(df) == 0:
            raise RuntimeError("DeepfakeEval2024Dataset: No audio files found.")
        self.rows = [
            (Path(row["full_path"]), 1 if row[label_col] == "real" else 0,
             Path(row[df.columns[0]]).name)
            for _, row in df.iterrows()
        ]
        print(f"[INFO] DeepfakeEval2024: loaded {len(self.rows)} samples.")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        audio_path, label_int, audio_name = self.rows[idx]
        waveform = self._process_audio(audio_path)
        return waveform, torch.tensor(label_int, dtype=torch.long), "deepfake_eval_2024", audio_name
        print(f"Source: {source}")
