from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from features_pixelwise import load_mammo_image, make_8ch_features, local_entropy_map, sobel_channels


@dataclass(frozen=True)
class PatchRef:
    row_idx: int
    y: int
    x: int


class MammoPixelwiseDataset(Dataset):
    """Patch dataset for weak pixel-wise VNN training.

    Expected CSV columns: image_path, cancer.
    If mask_path is present and points to a valid mask, it is used. Otherwise a weak pseudo-mask is created:
    cancer=0 -> all zeros; cancer=1 -> top gradient + entropy regions.
    """

    def __init__(
        self,
        csv_path: str | Path | pd.DataFrame,
        patch_size: int = 256,
        stride: int = 128,
        indices: list[int] | np.ndarray | None = None,
        cache_images: bool = False,
        max_patches_per_image: int | None = None,
    ):
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.reset_index(drop=True).copy()
        else:
            self.df = pd.read_csv(csv_path).reset_index(drop=True)
        if indices is not None:
            self.df = self.df.iloc[list(indices)].reset_index(drop=True)
        if "image_path" not in self.df.columns or "cancer" not in self.df.columns:
            raise ValueError("CSV/DataFrame must contain image_path and cancer columns")
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.cache_images = bool(cache_images)
        self.max_patches_per_image = max_patches_per_image
        self._image_cache: dict[int, np.ndarray] = {}
        self.read_errors: list[dict[str, str | int]] = []
        self.patch_index = self._build_patch_index()
        if self.read_errors:
            print(f"Skipped unreadable images: {len(self.read_errors)}")
            for error in self.read_errors[:5]:
                print(f"  skip {error['image_path']}: {error['error']}")
        if not self.patch_index:
            raise RuntimeError("No readable images available for patch indexing")

    def _load(self, row_idx: int) -> np.ndarray:
        if self.cache_images and row_idx in self._image_cache:
            return self._image_cache[row_idx]
        img = load_mammo_image(self.df.loc[row_idx, "image_path"])
        if self.cache_images:
            self._image_cache[row_idx] = img
        return img

    def _coords_for_shape(self, h: int, w: int) -> list[tuple[int, int]]:
        ps, st = self.patch_size, self.stride
        if h < ps or w < ps:
            return [(0, 0)]
        ys = list(range(0, h - ps + 1, st))
        xs = list(range(0, w - ps + 1, st))
        if ys[-1] != h - ps:
            ys.append(h - ps)
        if xs[-1] != w - ps:
            xs.append(w - ps)
        return [(y, x) for y in ys for x in xs]

    def _build_patch_index(self) -> list[PatchRef]:
        refs: list[PatchRef] = []
        for row_idx in tqdm(range(len(self.df)), desc="Indexing patches"):
            try:
                img = self._load(row_idx)
            except (OSError, IOError, ValueError, RuntimeError) as exc:
                self.read_errors.append({
                    "row_idx": row_idx,
                    "image_path": str(self.df.loc[row_idx, "image_path"]),
                    "error": repr(exc),
                })
                continue
            coords = self._coords_for_shape(*img.shape)
            if self.max_patches_per_image and len(coords) > self.max_patches_per_image:
                # Deterministic center-biased subset to keep training practical on very large mammograms.
                cy, cx = img.shape[0] / 2.0, img.shape[1] / 2.0
                coords = sorted(coords, key=lambda p: (p[0] + self.patch_size / 2 - cy) ** 2 + (p[1] + self.patch_size / 2 - cx) ** 2)
                coords = coords[: self.max_patches_per_image]
            refs.extend(PatchRef(row_idx, y, x) for y, x in coords)
        return refs

    def __len__(self) -> int:
        return len(self.patch_index)

    def _extract_patch(self, img: np.ndarray, y: int, x: int) -> np.ndarray:
        ps = self.patch_size
        if img.shape[0] < ps or img.shape[1] < ps:
            return cv2.resize(img, (ps, ps), interpolation=cv2.INTER_AREA)
        return img[y : y + ps, x : x + ps]

    def _real_mask_patch(self, row: pd.Series, y: int, x: int) -> np.ndarray | None:
        mask_path = row.get("mask_path", None)
        if not isinstance(mask_path, str) or not mask_path:
            return None
        p = Path(mask_path)
        if not p.exists():
            return None
        mask = load_mammo_image(p) > 0.5
        ps = self.patch_size
        if mask.shape[0] < ps or mask.shape[1] < ps:
            return cv2.resize(mask.astype(np.uint8), (ps, ps), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask[y : y + ps, x : x + ps].astype(bool)

    def _pseudo_mask(self, patch: np.ndarray, cancer: int) -> np.ndarray:
        if int(cancer) == 0:
            return np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        _, _, grad = sobel_channels(patch)
        ent = local_entropy_map(patch)
        score = 0.60 * grad + 0.40 * ent
        thresh = float(np.quantile(score, 0.92))
        mask = score >= thresh
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return mask.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.patch_index[idx]
        row = self.df.loc[ref.row_idx]
        img = self._load(ref.row_idx)
        patch = self._extract_patch(img, ref.y, ref.x)
        features = make_8ch_features(patch, self.patch_size)
        real_mask = self._real_mask_patch(row, ref.y, ref.x)
        if real_mask is None:
            mask = self._pseudo_mask(patch, int(row["cancer"]))
        else:
            mask = real_mask.astype(np.float32)
        mask_t = torch.from_numpy(mask[None, ...].astype(np.float32))
        label = torch.tensor(float(row["cancer"]), dtype=torch.float32)
        return features, mask_t, label
