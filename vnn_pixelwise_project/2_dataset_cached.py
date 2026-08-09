from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedMammoPatchDataset(Dataset):
    """Dataset reading only precomputed .pt tensors; no DICOM or feature computation."""

    def __init__(self, manifest: str | Path | pd.DataFrame, image_paths: set[str] | None = None, return_metadata: bool = False):
        self.df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        if image_paths is not None:
            self.df = self.df[self.df["image_path"].isin(image_paths)].copy()
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError("Cached dataset is empty")
        self.return_metadata = return_metadata
        image_codes, _ = pd.factorize(self.df["image_path"], sort=True)
        self.df["image_code"] = image_codes.astype("int64")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        item = torch.load(row["pt_path"], map_location="cpu", weights_only=False)
        x = torch.nan_to_num(item["x"].float(), nan=0.0, posinf=1.0, neginf=0.0)
        y = torch.tensor(float(row["cancer"]), dtype=torch.float32)
        if not self.return_metadata:
            return x, y
        return x, y, torch.tensor(int(row["image_code"]), dtype=torch.long)
