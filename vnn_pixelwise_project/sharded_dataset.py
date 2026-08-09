from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


class ShardedMILDataset(Dataset):
    """MIL bags loaded from one cached shard at a time; never reads DICOM or computes features."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str | None = None,
        mmap: bool = True,
        channels: tuple[int, ...] | None = None,
        training: bool = False,
        augment: bool = False,
        feature_engineering: str = "none",
        bag_drop_rate: float = 0.0,
        channel_drop_rate: float = 0.0,
        noise_std: float = 0.0,
    ):
        self.df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No images in sharded dataset for split={split!r}.")
        self.use_mmap = mmap
        self.channels = channels
        self.training = training
        self.augment = augment and training
        self.feature_engineering = feature_engineering
        self.bag_drop_rate = float(bag_drop_rate)
        self.channel_drop_rate = float(channel_drop_rate)
        self.noise_std = float(noise_std)
        self._loaded_path: str | None = None
        self._shard: dict | None = None

    def _engineer_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_engineering != "stats" or x.shape[1] < 2:
            return x
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        return torch.cat([x, mean, std], dim=1)

    def _augment_bag(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return x, valid
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-2,))
        if torch.rand(()) < 0.35:
            turns = int(torch.randint(1, 4, (1,)).item())
            x = torch.rot90(x, turns, dims=(-2, -1))
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.channel_drop_rate > 0 and x.shape[1] > 1:
            channel_mask = torch.rand(x.shape[1]) < self.channel_drop_rate
            if channel_mask.any() and not channel_mask.all():
                x = x.clone()
                x[:, channel_mask] = 0
        if self.bag_drop_rate > 0 and valid.any():
            bag_mask = torch.rand(valid.shape[0]) < self.bag_drop_rate
            valid = valid & ~bag_mask
            if not valid.any():
                valid = valid.clone()
                valid[int(torch.randint(0, len(valid), (1,)).item())] = True
        return x, valid

    def __len__(self) -> int:
        return len(self.df)

    def _load_shard(self, path: str) -> dict:
        if path != self._loaded_path:
            self._shard = None
            gc.collect()
            kwargs = {"map_location": "cpu", "weights_only": False}
            if self.use_mmap:
                kwargs["mmap"] = True
            try:
                self._shard = torch.load(path, **kwargs)
            except (RuntimeError, TypeError):
                kwargs.pop("mmap", None)
                self._shard = torch.load(path, **kwargs)
            self._loaded_path = path
        assert self._shard is not None
        return self._shard

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        shard = self._load_shard(str(row["shard_path"]))
        start = int(row["start"])
        end = start + int(row["count"])
        # Keep precomputed float16 tensors compact through CPU loading and PCIe transfer;
        # CUDA AMP handles compute precision in the training loop.
        x = shard["x"][start:end]
        if self.channels is not None:
            x = x[:, self.channels]
        valid = shard.get("valid_patch", torch.ones(len(shard["y"]), dtype=torch.bool))[start:end].bool()
        if self.training and (self.augment or self.feature_engineering != "none"):
            x = x.float()
        x = self._engineer_features(x)
        x, valid = self._augment_bag(x, valid)
        metadata = {
            "patient_id": str(row["patient_id"]),
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "view": str(row["view"]),
            "density": str(row["density"]),
            "laterality": str(row.get("laterality", "")),
            "difficult_negative_case": _as_bool(row["difficult_negative_case"]),
            "selection_score": float(row["selection_score"]) if "selection_score" in row and not pd.isna(row["selection_score"]) else 0.5,
            "selection_weight": float(row["selection_weight"]) if "selection_weight" in row and not pd.isna(row["selection_weight"]) else 1.0,
            "coords": shard["coords"][start:end].long(),
            "shard_id": int(row["shard_id"]),
        }
        return x, torch.tensor(float(row["y"]), dtype=torch.float32), valid, metadata


class ShardBatchSampler(Sampler[list[int]]):
    """Yield batches within a shard so each worker reuses its currently loaded shard."""

    def __init__(self, dataset: ShardedMILDataset, batch_size: int, shuffle: bool, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for idx, shard_id in enumerate(dataset.df["shard_id"].astype(int)):
            grouped[int(shard_id)].append(idx)
        self.groups = grouped

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shard_ids = list(self.groups)
        if self.shuffle:
            rng.shuffle(shard_ids)
        for shard_id in shard_ids:
            indices = self.groups[shard_id].copy()
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size]

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.groups.values())
