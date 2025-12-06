from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from recsys_transformer_seq.config import PROJECT_ROOT


class SequenceDataset(Dataset):
    """(sequence, target) pairs for next-item prediction."""

    def __init__(self, csv_path: Path, max_len: int):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing sequences CSV at {csv_path}")

        df = pd.read_csv(csv_path)
        self.user_ids = df["userId"].values
        self.targets_raw = df["target"].values
        self.sequences_raw = df["sequence"].values
        self.max_len = max_len

        # Build item vocabulary
        all_items = set(self.targets_raw)
        for seq_str in self.sequences_raw:
            all_items.update(map(int, seq_str.split()))

        self.item2idx: Dict[int, int] = {
            item_id: idx + 1 for idx, item_id in enumerate(sorted(all_items))
        }  # 0 = padding
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.num_items = len(self.item2idx) + 1

    def __len__(self) -> int:
        return len(self.targets_raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_str = self.sequences_raw[idx]
        target_raw = int(self.targets_raw[idx])

        seq_ids = [self.item2idx[int(i)] for i in seq_str.split()]
        target_id = self.item2idx[target_raw]

        if len(seq_ids) >= self.max_len:
            seq_ids = seq_ids[-self.max_len :]
        else:
            seq_ids = [0] * (self.max_len - len(seq_ids)) + seq_ids

        seq_tensor = torch.tensor(seq_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_id, dtype=torch.long)
        return seq_tensor, target_tensor


def build_dataloader(
    csv_path: Path,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
):
    ds = SequenceDataset(csv_path, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl, ds.num_items