#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analisar_alizams_cores_dicom_csv import (
    COLOR_BINS, apply_lut, breast_mask, color_metrics, load_lut, lut_profile, resize_image,
)
from features_pixelwise import read_dicom_gray_with_metadata


def parse_args() -> argparse.Namespace:
    root = Path("/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25")
    p = argparse.ArgumentParser(description="Prepare real RSNA batches for CUDA Fortran AlizaMS benchmark.")
    p.add_argument("--csv", default=str(root / "rsna_kaggle_oficial/train.csv"))
    p.add_argument("--images-dir", default=str(root / "rsna_kaggle_oficial/train_images"))
    p.add_argument("--lut-file", default=str(root / "iaimgem/sistema_integrado_rsna_csv_treino/alizams_rainbowb_cores_todas_imagens/alizams_luts.h"))
    p.add_argument("--real-images", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--out", default="alizams_cuda_fortran_input.bin")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv, dtype={"patient_id": str, "image_id": str}).head(args.real_images)
    lut = load_lut(Path(args.lut_file))
    gray_images, masks, reference = [], [], []
    for row in frame.itertuples(index=False):
        gray, _metadata = read_dicom_gray_with_metadata(Path(args.images_dir) / row.patient_id / f"{row.image_id}.dcm")
        gray = resize_image(gray, args.image_size)
        mask = breast_mask(gray)
        values, _first, _second = color_metrics(apply_lut(gray, lut), mask)
        reference.append([values["hue_medio_graus"], values["saturacao_media"], values["brilho_medio"],
                          *[values[f"freq_{name}"] for name, _hex in COLOR_BINS]])
        gray_images.append(gray.astype(np.float32))
        masks.append(mask.astype(np.int32))
    repeats = np.arange(args.batch_size) % len(gray_images)
    gray_batch = np.stack(gray_images)[repeats].astype(np.float32)
    mask_batch = np.stack(masks)[repeats].astype(np.int32)
    reference_batch = np.asarray(reference, dtype=np.float32)[repeats]
    bins, hue, sat, val, weight = lut_profile(lut)
    with Path(args.out).open("wb") as stream:
        np.asarray([args.batch_size, args.image_size, args.image_size, len(lut)], dtype=np.int32).tofile(stream)
        gray_batch.tofile(stream)
        mask_batch.tofile(stream)
        bins.astype(np.int32).tofile(stream)
        hue.astype(np.float32).tofile(stream)
        sat.astype(np.float32).tofile(stream)
        val.astype(np.float32).tofile(stream)
        weight.astype(np.float32).tofile(stream)
        reference_batch.tofile(stream)
    print(f"saved={args.out} batch={args.batch_size} shape={args.image_size}x{args.image_size} lut={len(lut)}")


if __name__ == "__main__":
    main()
