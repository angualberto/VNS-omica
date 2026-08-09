#!/usr/bin/env python3
"""
Compara preprocessamentos para segmentacao de mama com mascaras oficiais.

Entrada esperada:
  dataset_index.csv com colunas como img/img2ch/mask/id.
  Imagens em .npy e mascaras em .png ou .npy.

O teste treina uma U-Net pequena em CUDA, quando disponivel, e salva um CSV
comparando Dice, IoU e pixel accuracy para cada variante.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_CANDIDATES = [
    Path("/media/angualberto/3524f135-7da6-4174-9260-79177dd75871/backup_notebook/Documentos/posteruff/dataset_mammo_oficial"),
    Path("/media/andre/3524f135-7da6-4174-9260-79177dd75871/backup_notebook/Documentos/posteruff/dataset_mammo_oficial"),
    Path("/home/angualberto/Documentos/posteruff/dataset_mammo_oficial"),
    Path("/home/andre/Documentos/posteruff/dataset_mammo_oficial"),
]


def find_dataset(dataset_dir: str | None) -> Path:
    if dataset_dir:
        p = Path(dataset_dir).expanduser()
        if (p / "dataset_index.csv").exists():
            return p
        raise FileNotFoundError(f"Nao encontrei dataset_index.csv em {p}")

    for p in DEFAULT_DATASET_CANDIDATES:
        if (p / "dataset_index.csv").exists():
            return p

    roots = [Path("/home/angualberto"), Path("/media/angualberto")]
    for root in roots:
        if not root.exists():
            continue
        for csv_path in root.rglob("dataset_index.csv"):
            if "dataset_mammo" in str(csv_path.parent).lower():
                return csv_path.parent

    raise FileNotFoundError(
        "Nao encontrei dataset_mammo_oficial com dataset_index.csv. "
        "Monte o HD correto ou passe --dataset-dir CAMINHO."
    )


def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)
    lo = float(np.nanmin(img[finite]))
    hi = float(np.nanmax(img[finite]))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def highpass_fft(img: np.ndarray, keep_radius_frac: float = 0.08) -> np.ndarray:
    img = normalize01(img)
    h, w = img.shape
    f = np.fft.fftshift(np.fft.fft2(img))
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = max(1, int(min(h, w) * keep_radius_frac))
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) >= radius**2
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))
    return normalize01(out)


def lowpass_blur(img: np.ndarray) -> np.ndarray:
    img = normalize01(img)
    return cv2.GaussianBlur(img, (7, 7), 0)


def dominant_frequency_topology(img: np.ndarray, out_size: int) -> np.ndarray:
    img = normalize01(img)
    hp = highpass_fft(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = normalize01(np.sqrt(gx * gx + gy * gy))
    lap = normalize01(np.abs(cv2.Laplacian(img, cv2.CV_32F, ksize=3)))
    topo = normalize01(0.55 * hp + 0.30 * grad + 0.15 * lap)
    return cv2.resize(topo, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def load_mask(path: Path, img_size: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Falha ao ler mascara: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.float32)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask[None, :, :]


def load_image_base(dataset_dir: Path, row: pd.Series, img_size: int) -> np.ndarray:
    img_rel = row.get("img")
    img2ch_rel = row.get("img2ch")
    if isinstance(img_rel, str) and img_rel:
        arr = np.load(dataset_dir / img_rel)
        if arr.ndim == 3:
            arr = arr[..., 0]
    elif isinstance(img2ch_rel, str) and img2ch_rel:
        arr = np.load(dataset_dir / img2ch_rel)
        arr = arr[..., 0] if arr.ndim == 3 else arr
    else:
        raise KeyError("dataset_index.csv precisa ter coluna img ou img2ch")

    arr = normalize01(arr)
    if arr.shape != (img_size, img_size):
        arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return arr


def make_channels(base: np.ndarray, variant: str, img_size: int) -> np.ndarray:
    base = normalize01(base)
    low = lowpass_blur(base)
    high = highpass_fft(base)
    topo = dominant_frequency_topology(base, img_size)

    if variant == "sem_filtro":
        chans = [base, base]
    elif variant == "passa_baixa_mais_alta":
        chans = [low, high]
    elif variant == "sem_passa_baixa":
        chans = [base, high]
    elif variant == "alta_mais_topologia":
        chans = [high, topo]
    else:
        raise ValueError(f"Variante desconhecida: {variant}")
    return np.stack(chans, axis=0).astype(np.float32)


class MammoMaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: Path, variant: str, img_size: int, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.variant = variant
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        base = load_image_base(self.dataset_dir, row, self.img_size)
        mask_rel = row.get("mask")
        if not isinstance(mask_rel, str) or not mask_rel:
            raise KeyError("dataset_index.csv precisa ter coluna mask")
        mask = load_mask(self.dataset_dir / mask_rel, self.img_size)
        img = make_channels(base, self.variant, self.img_size)

        if self.augment:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy()
                mask = mask[:, :, ::-1].copy()
            if random.random() < 0.2:
                img = img[:, ::-1, :].copy()
                mask = mask[:, ::-1, :].copy()

        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.float32))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 24):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.b = DoubleConv(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(F.max_pool2d(c1, 2))
        c3 = self.d3(F.max_pool2d(c2, 2))
        b = self.b(F.max_pool2d(c3, 2))
        x = self.u3(b)
        x = self.c3(torch.cat([x, c3], dim=1))
        x = self.u2(x)
        x = self.c2(torch.cat([x, c2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x, c1], dim=1))
        return self.out(x)


def dice_loss(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (den + smooth)).mean()


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total_loss = 0.0
    total = 0
    inter = union = pred_sum = target_sum = correct = pixels = 0.0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        inter += float((pred * y).sum().item())
        union += float(((pred + y) > 0).float().sum().item())
        pred_sum += float(pred.sum().item())
        target_sum += float(y.sum().item())
        correct += float((pred == y).float().sum().item())
        pixels += float(y.numel())
    dice = (2.0 * inter + 1.0) / (pred_sum + target_sum + 1.0)
    iou = (inter + 1.0) / (union + 1.0)
    return {
        "loss": total_loss / max(total, 1),
        "dice": dice,
        "iou": iou,
        "pixel_acc": correct / max(pixels, 1.0),
    }


def train_variant(args, df_train, df_val, df_test, dataset_dir, out_dir, variant, device):
    train_ds = MammoMaskDataset(df_train, dataset_dir, variant, args.img_size, augment=True)
    val_ds = MammoMaskDataset(df_val, dataset_dir, variant, args.img_size, augment=False)
    test_ds = MammoMaskDataset(df_test, dataset_dir, variant, args.img_size, augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    model = UNetSmall(in_ch=2, base=args.base_filters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val = -math.inf
    best_path = out_dir / f"unet_{variant}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        val_metrics = evaluate(model, val_dl, device)
        print(f"{variant} epoch {epoch}/{args.epochs}: val_iou={val_metrics['iou']:.4f} val_dice={val_metrics['dice']:.4f}")
        if val_metrics["iou"] > best_val:
            best_val = val_metrics["iou"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_dl, device)
    test_metrics.update({"variant": variant, "model_path": str(best_path), "device": str(device)})
    return test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--out-dir", default="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem/sistema_integrado/unet_mascaras_oficiais")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-filters", type=int, default=24)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=["sem_filtro", "passa_baixa_mais_alta", "sem_passa_baixa", "alta_mais_topologia"])
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset_dir = find_dataset(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_dir / "dataset_index.csv")
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)

    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1 / 3, random_state=42)
    df_train, df_val, df_test = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {dataset_dir}")
    print(f"Treino={len(df_train)} Validacao={len(df_val)} Teste={len(df_test)}")
    print(f"Device: {device}")

    rows = []
    for variant in args.variants:
        rows.append(train_variant(args, df_train, df_val, df_test, dataset_dir, out_dir, variant, device))

    result_csv = out_dir / "comparacao_preprocessamento_unet_mascaras_oficiais.csv"
    pd.DataFrame(rows).sort_values("iou", ascending=False).to_csv(result_csv, index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "n_total": int(len(df)),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "device": str(device),
        "result_csv": str(result_csv),
    }
    (out_dir / "resumo_teste_unet_mascaras_oficiais.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nResultado salvo em:", result_csv)
    print(pd.read_csv(result_csv).to_string(index=False))


if __name__ == "__main__":
    main()
