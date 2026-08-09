#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analisar_alizams_cores_dicom_csv import (
    COLOR_BINS,
    apply_lut,
    breast_mask,
    color_metrics,
    load_lut,
    lut_profile,
    resize_image,
)
from features_pixelwise import read_dicom_gray_with_metadata


def parse_args() -> argparse.Namespace:
    root = Path("/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25")
    project = root / "iaimgem"
    p = argparse.ArgumentParser(description="Validate and benchmark AlizaMS LUT color metrics on CUDA batches.")
    p.add_argument("--csv", default=str(root / "rsna_kaggle_oficial/train.csv"))
    p.add_argument("--images-dir", default=str(root / "rsna_kaggle_oficial/train_images"))
    p.add_argument("--lut-file", default=str(project / "sistema_integrado_rsna_csv_treino/alizams_rainbowb_cores_todas_imagens/alizams_luts.h"))
    p.add_argument("--images", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--image-size", type=int, default=256)
    return p.parse_args()


def cuda_color_metrics(gray: torch.Tensor, mask: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
    bins, hue, sat, val, weight = lut_profile(lut)
    device = gray.device
    lookup_bin = torch.as_tensor(bins - 1, dtype=torch.int64, device=device)
    lookup_hue = torch.as_tensor(hue, dtype=torch.float32, device=device)
    lookup_sat = torch.as_tensor(sat, dtype=torch.float32, device=device)
    lookup_val = torch.as_tensor(val, dtype=torch.float32, device=device)
    lookup_weight = torch.as_tensor(weight, dtype=torch.float32, device=device)
    index = torch.clamp(torch.floor(gray * len(lut)).long(), 0, len(lut) - 1)
    color_bin = lookup_bin[index].flatten(1)
    pixel_weight = (lookup_weight[index] * mask).flatten(1)
    frequencies = torch.zeros((gray.shape[0], 8), dtype=torch.float32, device=device)
    frequencies.scatter_add_(1, color_bin, pixel_weight)
    denominator = pixel_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
    frequencies = frequencies / denominator
    hue_mean = (lookup_hue[index] * mask * lookup_weight[index]).flatten(1).sum(1, keepdim=True) / denominator
    sat_mean = (lookup_sat[index] * mask * lookup_weight[index]).flatten(1).sum(1, keepdim=True) / denominator
    val_mean = (lookup_val[index] * mask * lookup_weight[index]).flatten(1).sum(1, keepdim=True) / denominator
    return torch.cat([hue_mean, sat_mean, val_mean, frequencies], dim=1)


def load_real_batch(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    source = pd.read_csv(args.csv, dtype={"patient_id": str, "image_id": str})
    images, masks = [], []
    for row in source.head(args.images).itertuples(index=False):
        path = Path(args.images_dir) / row.patient_id / f"{row.image_id}.dcm"
        gray, _meta = read_dicom_gray_with_metadata(path)
        gray = resize_image(gray, args.image_size)
        images.append(gray)
        masks.append(breast_mask(gray))
    return np.stack(images).astype(np.float32), np.stack(masks).astype(np.float32)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    device = torch.device("cuda")
    lut = load_lut(Path(args.lut_file))
    real_gray, real_mask = load_real_batch(args)
    cuda_result = cuda_color_metrics(
        torch.from_numpy(real_gray).to(device), torch.from_numpy(real_mask).to(device), lut
    ).cpu().numpy()
    cpu_result = []
    for gray, mask in zip(real_gray, real_mask):
        values, primary, _second = color_metrics(apply_lut(gray, lut), mask.astype(bool))
        cpu_result.append([values["hue_medio_graus"], values["saturacao_media"], values["brilho_medio"],
                           *[values[f"freq_{name}"] for name, _hex in COLOR_BINS]])
        gpu_primary = COLOR_BINS[int(cuda_result[len(cpu_result) - 1, 3:].argmax())][0]
        if gpu_primary != primary:
            raise RuntimeError(f"Dominant color mismatch: CPU={primary} CUDA={gpu_primary}")
    max_difference = float(np.max(np.abs(np.asarray(cpu_result) - cuda_result)))

    repeated_gray = np.resize(real_gray, (args.batch_size, args.image_size, args.image_size))
    repeated_mask = np.resize(real_mask, (args.batch_size, args.image_size, args.image_size))
    gray_gpu = torch.from_numpy(repeated_gray).to(device)
    mask_gpu = torch.from_numpy(repeated_mask).to(device)
    for _ in range(3):
        cuda_color_metrics(gray_gpu, mask_gpu, lut)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        cuda_color_metrics(gray_gpu, mask_gpu, lut)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total = args.batch_size * args.iterations
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda_runtime:", torch.version.cuda)
    print("images_validated:", len(real_gray))
    print("dominant_colors_equal: True")
    print("max_numeric_difference:", max_difference)
    print("batch_size:", args.batch_size)
    print("milliseconds_per_batch:", elapsed / args.iterations * 1000.0)
    print("images_per_second:", total / elapsed)
    print("peak_vram_mib:", torch.cuda.max_memory_allocated() / 1024**2)


if __name__ == "__main__":
    main()
