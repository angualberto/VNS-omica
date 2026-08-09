from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from features_pixelwise import load_mammo_image, make_8ch_features, read_dicom_gray_with_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute 8-channel mammography patches to .pt cache")
    p.add_argument("--csv", required=True, help="CSV containing image_path and cancer")
    p.add_argument("--out-dir", required=True, help="Directory for cached .pt patches and manifest")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--min-tissue-frac", type=float, default=0.03, help="Skip mostly empty patches")
    p.add_argument("--max-patches-per-image", type=int, default=0, help="0 keeps all valid patches")
    p.add_argument("--test-size", type=float, default=0.25, help="Used only when input CSV has no split column")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def patch_coords(h: int, w: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    if h < patch_size or w < patch_size:
        return [(0, 0)]
    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)
    return [(x, y) for y in ys for x in xs]


def extract_patch(img: np.ndarray, x0: int, y0: int, patch_size: int) -> np.ndarray:
    if img.shape[0] < patch_size or img.shape[1] < patch_size:
        return cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return img[y0:y0 + patch_size, x0:x0 + patch_size]


def selected_coords(img: np.ndarray, args: argparse.Namespace) -> list[tuple[int, int]]:
    scored = []
    for x0, y0 in patch_coords(*img.shape, args.patch_size, args.stride):
        patch = extract_patch(img, x0, y0, args.patch_size)
        tissue_frac = float((patch > 0.02).mean())
        if tissue_frac < args.min_tissue_frac:
            continue
        # Used only for optional limiting during offline cache generation.
        score = float(patch.std() + 0.15 * patch.mean())
        scored.append((score, x0, y0))
    if not scored:
        coords = patch_coords(*img.shape, args.patch_size, args.stride)
        return coords[:1]
    scored.sort(reverse=True)
    if args.max_patches_per_image > 0:
        scored = scored[:args.max_patches_per_image]
    return [(x0, y0) for _score, x0, y0 in scored]


def ensure_split(source: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    """Create split before patch computation, at image level, or preserve supplied split."""
    source = source.copy()
    if "split" in source.columns and {"train", "test"}.issubset(set(source["split"].dropna().astype(str))):
        return source
    unique_images = source[["image_path", "cancer"]].drop_duplicates("image_path")
    train_img, test_img = train_test_split(
        unique_images, test_size=test_size, stratify=unique_images["cancer"], random_state=seed
    )
    test_paths = set(test_img["image_path"])
    source["split"] = source["image_path"].map(lambda p: "test" if p in test_paths else "train")
    return source


def read_image_and_metadata(path: str) -> tuple[np.ndarray, dict[str, object]]:
    suffix = Path(path).suffix.lower()
    if suffix in {".dcm", ".dicom", ""}:
        return read_dicom_gray_with_metadata(path)
    return load_mammo_image(path), {"windowing_applied": False, "dicom": False}


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.csv)
    required = {"image_path", "cancer"}
    if not required.issubset(source.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    source = ensure_split(source, args.test_size, args.seed)
    out_dir = Path(args.out_dir)
    patch_dir = out_dir / "patches_pt"
    patch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    error_path = out_dir / "dicom_read_errors.csv"
    dtype = torch.float32
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    saved = 0

    for image_idx, row in tqdm(source.iterrows(), total=len(source), desc="Precomputing images"):
        image_path = str(row["image_path"])
        try:
            img, dicom_metadata = read_image_and_metadata(image_path)
            coords = selected_coords(img, args)
            for patch_id, (x0, y0) in enumerate(coords):
                patch = extract_patch(img, x0, y0, args.patch_size)
                x = make_8ch_features(patch, args.patch_size).to(dtype=dtype).contiguous()
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                pt_name = f"img_{int(image_idx):06d}_patch_{patch_id:04d}.pt"
                pt_path = patch_dir / pt_name
                if args.overwrite or not pt_path.exists():
                    torch.save({
                        "x": x.float(),
                        "y": torch.tensor(float(row["cancer"]), dtype=torch.float32),
                        "image_path": image_path,
                        "image_idx": int(image_idx),
                        "patch_id": int(patch_id),
                        "coords": (int(x0), int(y0)),
                        "split": str(row["split"]),
                        "dicom_metadata": dicom_metadata,
                        "params": {
                            "patch_size": int(args.patch_size),
                            "stride": int(args.stride),
                            "windowing": True,
                            "dicom": Path(image_path).suffix.lower() in {".dcm", ".dicom", ""},
                            "normalization": "dicom-window-then-minmax-[0,1]",
                            "channels": [
                                "gray", "gradient_magnitude", "sobel_x", "sobel_y",
                                "local_entropy", "fft_low", "fft_high", "pseudo_color_family"
                            ],
                        },
                    }, pt_path)
                item = {
                    "pt_path": str(pt_path),
                    "image_path": image_path,
                    "image_idx": int(image_idx),
                    "cancer": int(row["cancer"]),
                    "patch_id": int(patch_id),
                    "x0": int(x0),
                    "y0": int(y0),
                    "split": str(row["split"]),
                    "windowing_applied": bool(dicom_metadata.get("windowing_applied", False)),
                    "photometric_interpretation": str(dicom_metadata.get("photometric_interpretation", "")),
                }
                rows.append(item)
                saved += 1
        except Exception as exc:
            errors.append({"image_idx": int(image_idx), "image_path": image_path, "error": repr(exc)})
            tqdm.write(f"skip unreadable DICOM: {image_path}: {exc!r}")

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No cached patches were generated")
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(errors, columns=["image_idx", "image_path", "error"]).to_csv(error_path, index=False)
    info = {
        "input_csv": str(args.csv),
        "manifest": str(manifest_path),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "cached_patches": int(saved),
        "read_errors": int(len(errors)),
        "stored_dtype": "torch.float32",
        "split_created_before_precompute": True,
        "split_counts_images": source[["image_path", "split"]].drop_duplicates()["split"].value_counts().to_dict(),
        "params": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "windowing": True,
            "normalization": "dicom-window-then-minmax-[0,1]",
            "seed": args.seed,
        },
    }
    (out_dir / "precompute_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
