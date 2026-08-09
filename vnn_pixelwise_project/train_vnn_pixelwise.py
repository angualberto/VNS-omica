from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_mammo_pixelwise import MammoPixelwiseDataset
from vnn_pixelwise_mammo import VNNPixelMammo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train weak pixel-wise VNN for mammography patches")
    p.add_argument("--csv", required=True, help="CSV with image_path,cancer columns")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-patches-per-image", type=int, default=16)
    p.add_argument("--cache-images", action="store_true", help="Keep decoded images in RAM to avoid re-reading DICOM for each patch")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def image_scores_from_patches(patch_probs: list[float], patch_labels: list[int], patch_image_ids: list[int], top_k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    tmp = pd.DataFrame({"row_idx": patch_image_ids, "prob": patch_probs, "label": patch_labels})
    for row_idx, g in tmp.groupby("row_idx"):
        vals = np.sort(g["prob"].to_numpy())[::-1]
        score = float(vals[: min(top_k, len(vals))].mean())
        rows.append((int(row_idx), int(g["label"].iloc[0]), score))
    out = pd.DataFrame(rows, columns=["row_idx", "label", "score"]).sort_values("row_idx")
    return out["label"].to_numpy(), out["score"].to_numpy()


def _binary_metrics(prefix: str, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | list]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
        f"{prefix}_pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: str, out_dir: Path, epoch: int | None = None) -> dict[str, float | list]:
    model.eval()
    patch_probs, patch_labels, patch_image_ids = [], [], []
    cursor = 0
    with torch.no_grad():
        for features, masks, labels in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            prob_maps = torch.sigmoid(logits)
            # Patch risk: top-k mean pixels, robust to tiny pseudo-lesion spots.
            flat = prob_maps.flatten(1)
            k = max(1, min(1024, flat.shape[1] // 32))
            probs = flat.topk(k, dim=1).values.mean(dim=1).detach().cpu().numpy()
            labels_np = labels.numpy().astype(int)
            for j, prob in enumerate(probs):
                ref = loader.dataset.patch_index[cursor + j]
                patch_probs.append(float(prob))
                patch_labels.append(int(labels_np[j]))
                patch_image_ids.append(int(ref.row_idx))
            cursor += len(labels_np)
    patch_y = np.asarray(patch_labels)
    patch_score = np.asarray(patch_probs)
    image_y, image_score = image_scores_from_patches(patch_probs, patch_labels, patch_image_ids, top_k=8)
    metrics = {}
    metrics.update(_binary_metrics("patch", patch_y, patch_score))
    metrics.update(_binary_metrics("image", image_y, image_score))
    pd.DataFrame({"row_idx": patch_image_ids, "label": patch_y, "score": patch_score, "pred": (patch_score >= 0.5).astype(int)}).to_csv(out_dir / f"pred_patch_epoch_{epoch or 'final'}.csv", index=False)
    pd.DataFrame({"label": image_y, "score": image_score, "pred": (image_score >= 0.5).astype(int)}).to_csv(out_dir / f"pred_image_epoch_{epoch or 'final'}.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    df = pd.read_csv(args.csv)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.25, stratify=df["cancer"].to_numpy(), random_state=args.seed)
    train_ds = MammoPixelwiseDataset(df, patch_size=args.patch_size, stride=args.stride, indices=train_idx, cache_images=args.cache_images, max_patches_per_image=args.max_patches_per_image)
    test_ds = MammoPixelwiseDataset(df, patch_size=args.patch_size, stride=args.stride, indices=test_idx, cache_images=args.cache_images, max_patches_per_image=args.max_patches_per_image)
    pd.DataFrame(train_ds.read_errors).to_csv(out_dir / "dicom_read_errors_train.csv", index=False)
    pd.DataFrame(test_ds.read_errors).to_csv(out_dir / "dicom_read_errors_test.csv", index=False)
    print(f"patches train={len(train_ds)} test={len(test_ds)} | skipped train={len(train_ds.read_errors)} test={len(test_ds.read_errors)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = VNNPixelMammo(in_ch=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    history = []
    best_pr = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for features, masks, _labels in pbar:
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and device == "cuda")):
                logits = model(features)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * features.size(0)
            n += features.size(0)
            pbar.set_postfix(loss=total_loss / max(n, 1))
        metrics = evaluate(model, test_loader, device, out_dir, epoch=epoch)
        row = {"epoch": epoch, "loss": total_loss / max(n, 1), **metrics}
        history.append(row)
        print(row)
        if metrics["image_pr_auc"] > best_pr:
            best_pr = metrics["image_pr_auc"]
            torch.save(model.state_dict(), out_dir / "best_vnn_pixelwise.pt")
    torch.save(model.state_dict(), out_dir / "last_vnn_pixelwise.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    final_metrics = evaluate(model, test_loader, device, out_dir, epoch=None)
    (out_dir / "metrics_final.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print("final:", final_metrics)


if __name__ == "__main__":
    main()
