from __future__ import annotations

import argparse
import gc
import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from features_pixelwise import make_8ch_features, read_dicom_gray_with_metadata


@dataclass
class PreprocessConfig:
    patch_size: int
    top_k_patches: int
    shard_size: int
    min_tissue_fraction: float
    background_threshold: float
    background_padding: int
    stored_dtype: str
    seed: int
    test_size: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute RSNA DICOM mammograms into large MIL shards.")
    p.add_argument("--csv", required=True, help="RSNA train.csv or CSV with image_path and cancer.")
    p.add_argument("--images-dir", help="Folder containing patient_id/image_id.dcm when image_path is absent.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--top-k-patches", type=int, default=8, help="Fixed MIL bag size per image.")
    p.add_argument("--shard-size", type=int, default=1024, help="Maximum patches per shard; float16 is about 256 MiB at 128x128.")
    p.add_argument("--min-tissue-fraction", type=float, default=0.05)
    p.add_argument("--background-threshold", type=float, default=0.02)
    p.add_argument("--background-padding", type=int, default=16)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stored-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--workers", type=int, default=4, help="Parallel DICOM/feature workers; results are bounded in flight.")
    p.add_argument("--max-images", type=int, default=0, help="Smoke-test limit after patient split; 0 means all.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_source(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    required = {"patient_id", "image_id", "cancer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required RSNA columns: {sorted(missing)}")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["cancer"] = df["cancer"].astype(int)
    if "image_path" not in df.columns:
        if not args.images_dir:
            raise ValueError("--images-dir is required when CSV has no image_path column.")
        images_dir = Path(args.images_dir)
        df["image_path"] = df.apply(
            lambda r: str(images_dir / str(r["patient_id"]) / f"{r['image_id']}.dcm"), axis=1
        )
    for col, default in {
        "view": "UNKNOWN",
        "density": "UNKNOWN",
        "difficult_negative_case": False,
        "laterality": "",
    }.items():
        if col not in df.columns:
            df[col] = default
    df["view"] = df["view"].fillna("UNKNOWN").astype(str)
    df["density"] = df["density"].fillna("UNKNOWN").astype(str)
    df["difficult_negative_case"] = df["difficult_negative_case"].fillna(False).astype(bool)
    return split_by_patient(df, args.test_size, args.seed, args.max_images)


def split_by_patient(df: pd.DataFrame, test_size: float, seed: int, max_images: int) -> pd.DataFrame:
    if "split" in df.columns and {"train", "test"}.issubset(set(df["split"].dropna().astype(str))):
        by_patient = df.groupby("patient_id")["split"].nunique()
        if int(by_patient.max()) != 1:
            raise ValueError("Input split leaks patient_id between train and test.")
        out = df.copy()
    else:
        patients = df.groupby("patient_id", as_index=False)["cancer"].max()
        train_patients, test_patients = train_test_split(
            patients["patient_id"],
            test_size=test_size,
            stratify=patients["cancer"],
            random_state=seed,
        )
        test_set = set(test_patients.astype(str))
        out = df.copy()
        out["split"] = out["patient_id"].map(lambda value: "test" if value in test_set else "train")
    if max_images > 0:
        sampled = []
        for split, group in out.groupby("split"):
            n = min(len(group), max(2, int(round(max_images * len(group) / len(out)))))
            sampled.append(group.sample(n=n, random_state=seed))
        out = pd.concat(sampled, ignore_index=True)
    train = set(out.loc[out["split"] == "train", "patient_id"])
    test = set(out.loc[out["split"] == "test", "patient_id"])
    if train & test:
        raise RuntimeError("Patient leakage detected after split construction.")
    return out.sort_values(["split", "patient_id", "image_id"]).reset_index(drop=True)


def crop_black_background(img: np.ndarray, threshold: float, padding: int) -> tuple[np.ndarray, tuple[int, int]]:
    mask = img > threshold
    if not np.any(mask):
        return img, (0, 0)
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(img.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(img.shape[1], int(xs.max()) + padding + 1)
    return img[y0:y1, x0:x1], (x0, y0)


def candidate_coords(img: np.ndarray, size: int) -> list[tuple[int, int]]:
    if img.shape[0] <= size or img.shape[1] <= size:
        return [(0, 0)]
    stride = size
    ys = list(range(0, img.shape[0] - size + 1, stride))
    xs = list(range(0, img.shape[1] - size + 1, stride))
    if ys[-1] != img.shape[0] - size:
        ys.append(img.shape[0] - size)
    if xs[-1] != img.shape[1] - size:
        xs.append(img.shape[1] - size)
    return [(x, y) for y in ys for x in xs]


def patch_at(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    if img.shape[0] < size or img.shape[1] < size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img[y:y + size, x:x + size]


def select_top_patches(
    img: np.ndarray, config: PreprocessConfig
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[bool]]:
    scored = []
    for x, y in candidate_coords(img, config.patch_size):
        patch = patch_at(img, x, y, config.patch_size)
        tissue = float((patch > config.background_threshold).mean())
        if tissue < config.min_tissue_fraction:
            continue
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        score = float(np.std(patch) + 0.20 * np.mean(cv2.magnitude(gx, gy)))
        scored.append((score, x, y, patch))
    if not scored:
        scored = [(0.0, 0, 0, patch_at(img, 0, 0, config.patch_size))]
    scored.sort(key=lambda value: value[0], reverse=True)
    selected = scored[: config.top_k_patches]
    valid = [True] * len(selected)
    while len(selected) < config.top_k_patches:
        selected.append(selected[-1])
        valid.append(False)
    return [item[3] for item in selected], [(item[1], item[2]) for item in selected], valid


def make_mil_features(patch: np.ndarray, patch_size: int) -> torch.Tensor:
    """Build offline features with channel 8 encoding the requested purple/green family."""
    features = make_8ch_features(patch, patch_size)
    color_score = torch.clamp(0.78 * features[0] + 0.22 * features[1], 0.0, 1.0)
    features[7] = torch.where(color_score >= 0.5, 0.75, 0.25)
    return features


class ShardWriter:
    def __init__(self, out: Path, config: PreprocessConfig, overwrite: bool):
        self.out = out
        self.shards = out / "shards"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.overwrite = overwrite
        self.shard_id = 0
        self.x: list[torch.Tensor] = []
        self.fields: dict[str, list] = {
            "y": [], "patient_id": [], "image_id": [], "view": [], "density": [],
            "difficult_negative_case": [], "coords": [], "split": [], "valid_patch": [],
        }
        self.manifest: list[dict[str, object]] = []

    def add_image(self, x: list[torch.Tensor], row: pd.Series, coords: list[tuple[int, int]], valid: list[bool]) -> None:
        if self.x and len(self.x) + len(x) > self.config.shard_size:
            self.flush()
        start = len(self.x)
        self.x.extend(x)
        n = len(x)
        self.fields["y"].extend([float(row["cancer"])] * n)
        self.fields["patient_id"].extend([str(row["patient_id"])] * n)
        self.fields["image_id"].extend([str(row["image_id"])] * n)
        self.fields["view"].extend([str(row["view"])] * n)
        self.fields["density"].extend([str(row["density"])] * n)
        self.fields["difficult_negative_case"].extend([bool(row["difficult_negative_case"])] * n)
        self.fields["coords"].extend(coords)
        self.fields["split"].extend([str(row["split"])] * n)
        self.fields["valid_patch"].extend(valid)
        self.manifest.append({
            "shard_id": self.shard_id,
            "shard_path": str(self.shards / f"shard_{self.shard_id:04d}.pt"),
            "start": start,
            "count": n,
            "valid_patches": int(sum(valid)),
            "patient_id": str(row["patient_id"]),
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "y": int(row["cancer"]),
            "view": str(row["view"]),
            "density": str(row["density"]),
            "difficult_negative_case": bool(row["difficult_negative_case"]),
            "split": str(row["split"]),
        })

    def flush(self) -> None:
        if not self.x:
            return
        output = self.shards / f"shard_{self.shard_id:04d}.pt"
        if output.exists() and not self.overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite or choose another --out-dir.")
        dtype = torch.float16 if self.config.stored_dtype == "float16" else torch.float32
        payload = {
            "x": torch.stack(self.x).to(dtype=dtype),
            "y": torch.tensor(self.fields["y"], dtype=torch.float32),
            "patient_id": self.fields["patient_id"],
            "image_id": self.fields["image_id"],
            "view": self.fields["view"],
            "density": self.fields["density"],
            "difficult_negative_case": torch.tensor(self.fields["difficult_negative_case"], dtype=torch.bool),
            "coords": torch.tensor(self.fields["coords"], dtype=torch.int32),
            "split": self.fields["split"],
            "valid_patch": torch.tensor(self.fields["valid_patch"], dtype=torch.bool),
        }
        torch.save(payload, output)
        self.shard_id += 1
        self.x.clear()
        for value in self.fields.values():
            value.clear()
        del payload
        gc.collect()


def process_image(row: dict[str, object], config: PreprocessConfig):
    try:
        img, _dicom_meta = read_dicom_gray_with_metadata(str(row["image_path"]))
        img, origin = crop_black_background(img, config.background_threshold, config.background_padding)
        patches, coords, valid = select_top_patches(img, config)
        coords = [(x + origin[0], y + origin[1]) for x, y in coords]
        features = [make_mil_features(patch, config.patch_size) for patch in patches]
        return row, features, coords, valid, None
    except Exception as exc:
        return row, None, None, None, repr(exc)


def iter_processed_images(source: pd.DataFrame, config: PreprocessConfig, workers: int):
    records = iter(source.to_dict("records"))
    if workers <= 1:
        for row in records:
            yield process_image(row, config)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(workers * 2):
            try:
                pending.add(executor.submit(process_image, next(records), config))
            except StopIteration:
                break
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
                try:
                    pending.add(executor.submit(process_image, next(records), config))
                except StopIteration:
                    pass


def main() -> None:
    args = parse_args()
    if args.top_k_patches < 1:
        raise ValueError("--top-k-patches must be at least 1.")
    if args.shard_size < args.top_k_patches:
        raise ValueError("--shard-size must hold at least one complete image bag.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    config = PreprocessConfig(
        patch_size=args.patch_size,
        top_k_patches=args.top_k_patches,
        shard_size=args.shard_size,
        min_tissue_fraction=args.min_tissue_fraction,
        background_threshold=args.background_threshold,
        background_padding=args.background_padding,
        stored_dtype=args.stored_dtype,
        seed=args.seed,
        test_size=args.test_size,
    )
    source = build_source(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source.to_csv(out / "image_split.csv", index=False)
    writer = ShardWriter(out, config, args.overwrite)
    errors: list[dict[str, str]] = []
    print(f"Preprocess workers: {args.workers}")
    for row, features, coords, valid, error in tqdm(
        iter_processed_images(source, config, args.workers), total=len(source), desc="DICOM -> shards"
    ):
        if error is None:
            writer.add_image(features, row, coords, valid)
        else:
            errors.append({"image_path": str(row["image_path"]), "error": error})
    writer.flush()
    manifest = pd.DataFrame(writer.manifest)
    if manifest.empty:
        raise RuntimeError("No image was converted to shards.")
    manifest.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(errors, columns=["image_path", "error"]).to_csv(out / "dicom_read_errors.csv", index=False)
    patients_train = set(manifest.loc[manifest["split"] == "train", "patient_id"])
    patients_test = set(manifest.loc[manifest["split"] == "test", "patient_id"])
    info = {
        "input_csv": str(args.csv),
        "images_dir": str(args.images_dir),
        "manifest": str(out / "manifest.csv"),
        "shards_dir": str(writer.shards),
        "preprocessing": {
            **asdict(config),
            "dicom_only": True,
            "monochrome1_inverted": True,
            "windowing": "WindowCenter/WindowWidth when present",
            "normalization": "float32 [0,1] before feature extraction",
            "black_background_removed": True,
            "channels": [
                "gray", "gradient_magnitude", "sobel_x", "sobel_y", "local_entropy",
                "fft_low", "fft_high", "pseudo_color_green_purple_family",
            ],
            "supervision": "global cancer label only; no pseudo-mask ground truth",
        },
        "images": int(len(manifest)),
        "patches": int(manifest["count"].sum()),
        "shards": int(manifest["shard_id"].nunique()),
        "read_errors": int(len(errors)),
        "patient_split_leakage": bool(patients_train & patients_test),
        "split_images": manifest["split"].value_counts().to_dict(),
        "split_patients": manifest.groupby("split")["patient_id"].nunique().to_dict(),
    }
    if info["patient_split_leakage"]:
        raise RuntimeError("Patient leakage detected in generated manifest.")
    (out / "preprocess_params.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
