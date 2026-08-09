from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from sharded_dataset import ShardBatchSampler, ShardedMILDataset


class VolterraBlock(nn.Module):
    """Low-rank quadratic VNN block after spatial reduction."""

    def __init__(self, in_ch: int, out_ch: int, rank: int, stride: int = 1):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.linear = nn.Conv2d(in_ch, out_ch, 1)
        self.u = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.v = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.quad = nn.Conv2d(rank, out_ch, 1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        return F.gelu(self.norm(self.linear(x) + self.quad(self.u(x) * self.v(x))))


class VNNPatchEncoder(nn.Module):
    def __init__(self, in_ch: int = 8, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            VolterraBlock(24, 32, rank=16, stride=2),
            VolterraBlock(32, 64, rank=24, stride=2),
            VolterraBlock(64, 96, rank=32, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(96, embedding_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(patches).flatten(1))


class VNNMILAttention(nn.Module):
    """VNN patch encoder with gated attention and one global image score."""

    def __init__(self, in_ch: int = 8, embedding_dim: int = 128, attention_dim: int = 64):
        super().__init__()
        self.encoder = VNNPatchEncoder(in_ch, embedding_dim)
        self.attention_v = nn.Linear(embedding_dim, attention_dim)
        self.attention_u = nn.Linear(embedding_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patches, channels, height, width = x.shape
        h = self.encoder(x.reshape(batch * patches, channels, height, width)).reshape(batch, patches, -1)
        attention = self.attention_w(torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))).squeeze(-1)
        if valid is not None:
            attention = attention.masked_fill(~valid, torch.finfo(attention.dtype).min)
        weights = torch.softmax(attention, dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled).squeeze(1), weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CUDA VNN-MIL from precomputed mammography shards.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=0, help="0 autotunes candidates 16,32,64.")
    p.add_argument("--batch-candidates", default="16,32,64")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--attention-top-k", type=int, default=3)
    p.add_argument("--feature-engineering", choices=["none", "stats"], default="stats")
    p.add_argument("--bag-drop-rate", type=float, default=0.10)
    p.add_argument("--channel-drop-rate", type=float, default=0.08)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--loss-type", choices=["bce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--selection-weight-floor", type=float, default=0.75)
    p.add_argument("--selection-weight-ceil", type=float, default=1.35)
    return p.parse_args()


def make_loader(
    dataset: ShardedMILDataset,
    batch_size: int,
    workers: int,
    prefetch: int,
    shuffle: bool,
    seed: int,
    cuda: bool,
) -> tuple[DataLoader, ShardBatchSampler]:
    sampler = ShardBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
    kwargs = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": cuda,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
    return DataLoader(dataset, **kwargs), sampler


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    pred = (score >= 0.5).astype(np.int64)
    multi_class = len(np.unique(y)) > 1
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, score)) if multi_class else None,
        "pr_auc": float(average_precision_score(y, score)) if multi_class else None,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def group_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    def evaluate_group(frame: pd.DataFrame) -> dict[str, object]:
        return binary_metrics(frame["y"].to_numpy(dtype=int), frame["score"].to_numpy(dtype=float))

    metrics: dict[str, object] = {"overall": evaluate_group(predictions)}
    metrics["by_view"] = {
        str(key): evaluate_group(value) for key, value in predictions.groupby("view", dropna=False)
    }
    metrics["by_density"] = {
        str(key): evaluate_group(value) for key, value in predictions.groupby("density", dropna=False)
    }
    metrics["difficult_negative_case"] = {
        "included": evaluate_group(predictions),
        "excluded": evaluate_group(predictions[~predictions["difficult_negative_case"]])
        if (~predictions["difficult_negative_case"]).any() else None,
        "only_difficult": evaluate_group(predictions[predictions["difficult_negative_case"]])
        if predictions["difficult_negative_case"].any() else None,
    }
    return metrics


def attach_selection_weights(manifest: pd.DataFrame, manifest_path: str, floor: float, ceil: float) -> pd.DataFrame:
    selected_path = Path(manifest_path).with_name("selected_patches_manifest.csv")
    if selected_path.exists():
        selected = pd.read_csv(selected_path, dtype={"patient_id": str, "image_id": str})
        score_column = "selection_weight" if "selection_weight" in selected.columns else "final_score"
        if {"patient_id", "image_id", score_column}.issubset(selected.columns):
            summary = selected.groupby(["patient_id", "image_id"], as_index=False).agg(
                selection_score=(score_column, "mean"),
                selection_score_max=(score_column, "max"),
                selection_patch_coverage=("valid_patch", "mean") if "valid_patch" in selected.columns else ("final_score", "size"),
            )
            manifest = manifest.merge(summary, on=["patient_id", "image_id"], how="left")
    if "selection_score" not in manifest.columns:
        manifest["selection_score"] = np.nan
    if "selection_score_max" not in manifest.columns:
        manifest["selection_score_max"] = np.nan
    if "selection_patch_coverage" not in manifest.columns:
        manifest["selection_patch_coverage"] = np.nan
    train_scores = manifest.loc[manifest["split"].astype(str) == "train", "selection_score"].dropna()
    if train_scores.empty:
        manifest["selection_score"] = manifest["selection_score"].fillna(0.5)
        manifest["selection_weight"] = 1.0
        return manifest
    low, high = float(train_scores.min()), float(train_scores.max())
    if high <= low:
        normalized = pd.Series(0.5, index=manifest.index)
    else:
        normalized = (manifest["selection_score"].fillna(train_scores.median()) - low) / (high - low)
    normalized = normalized.clip(0.0, 1.0)
    manifest["selection_score"] = manifest["selection_score"].fillna(train_scores.median())
    manifest["selection_weight"] = floor + (ceil - floor) * normalized
    manifest["selection_weight"] = manifest["selection_weight"].clip(floor, ceil)
    manifest["selection_patch_coverage"] = manifest["selection_patch_coverage"].fillna(1.0)
    return manifest


def build_sample_weights(metadata: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    weights = [float(metadata_value(metadata, "selection_weight", i)) if "selection_weight" in metadata else 1.0 for i in range(batch_size)]
    return torch.tensor(weights, device=device, dtype=torch.float32)


def weighted_loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    pos_weight: torch.Tensor,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    target = target.float()
    if label_smoothing > 0:
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
    base = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    if loss_type == "focal":
        probabilities = torch.sigmoid(logits)
        pt = target * probabilities + (1.0 - target) * (1.0 - probabilities)
        base = base * (1.0 - pt).pow(focal_gamma)
    base = base * sample_weight
    return base.mean()


def metadata_value(metadata: dict, key: str, i: int):
    value = metadata[key]
    if torch.is_tensor(value):
        return value[i].item()
    return value[i]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for x, y, valid, metadata in loader:
            x = x.to(device, non_blocking=True)
            if device.type != "cuda":
                x = x.float()
            valid_gpu = valid.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, attention = model(x, valid_gpu)
            scores = torch.sigmoid(logits).cpu().numpy()
            attention = attention.cpu()
            for i in range(len(y)):
                row = {
                    "patient_id": str(metadata_value(metadata, "patient_id", i)),
                    "image_id": str(metadata_value(metadata, "image_id", i)),
                    "image_path": str(metadata_value(metadata, "image_path", i)),
                    "view": str(metadata_value(metadata, "view", i)),
                    "density": str(metadata_value(metadata, "density", i)),
                    "difficult_negative_case": bool(metadata_value(metadata, "difficult_negative_case", i)),
                    "y": int(y[i].item()),
                    "score": float(scores[i]),
                }
                rows.append(row)
                valid_idx = torch.where(valid[i])[0]
                ranked = valid_idx[torch.argsort(attention[i, valid_idx], descending=True)[:top_k]]
                for rank, patch_idx in enumerate(ranked.tolist(), start=1):
                    coords = metadata["coords"][i, patch_idx]
                    attention_rows.append({
                        **row,
                        "rank": rank,
                        "patch_index": int(patch_idx),
                        "x0": int(coords[0].item()),
                        "y0": int(coords[1].item()),
                        "attention": float(attention[i, patch_idx].item()),
                    })
    predictions = pd.DataFrame(rows)
    predictions["pred"] = (predictions["score"] >= 0.5).astype(int)
    return predictions, pd.DataFrame(attention_rows)


def autotune_batch(
    dataset: ShardedMILDataset,
    candidates: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    if device.type != "cuda":
        return candidates[0]
    selected = candidates[0]
    print("Batch-size autotune:")
    for batch_size in candidates:
        model = VNNMILAttention().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loader, _ = make_loader(dataset, batch_size, 0, args.prefetch_factor, False, args.seed, True)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x, y, valid, _ = next(iter(loader))
            start = time.perf_counter()
            x = x.to(device, non_blocking=True)
            if device.type != "cuda":
                x = x.float()
            y = y.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits, _ = model(x, valid)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_reserved() / 1024**3
            patches = int(x.shape[0] * x.shape[1])
            print(f"  batch={batch_size:<3} ok time={seconds:.4f}s patches/s={patches/seconds:.1f} peak_vram={peak:.2f} GB")
            selected = batch_size
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch_size:<3} CUDA OOM")
            break
        finally:
            del model, optimizer, loader
            gc.collect()
            torch.cuda.empty_cache()
    return selected


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        print("CUDA unavailable; training will run on CPU.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = attach_selection_weights(pd.read_csv(args.manifest), args.manifest, args.selection_weight_floor, args.selection_weight_ceil)
    if "split" not in manifest.columns or not {"train", "test"}.issubset(set(manifest["split"].astype(str))):
        raise ValueError("Manifest must contain train/test split created before preprocessing.")
    train_patients = set(manifest.loc[manifest["split"] == "train", "patient_id"].astype(str))
    test_patients = set(manifest.loc[manifest["split"] == "test", "patient_id"].astype(str))
    if train_patients & test_patients:
        raise ValueError("Patient leakage detected: patient_id appears in train and test shards.")
    feature_engineering = args.feature_engineering
    in_ch = 8 + 2 if feature_engineering == "stats" else 8
    train_ds = ShardedMILDataset(
        manifest,
        split="train",
        mmap=not args.no_mmap,
        training=True,
        augment=True,
        feature_engineering=feature_engineering,
        bag_drop_rate=args.bag_drop_rate,
        channel_drop_rate=args.channel_drop_rate,
        noise_std=args.noise_std,
    )
    test_ds = ShardedMILDataset(
        manifest,
        split="test",
        mmap=not args.no_mmap,
        training=False,
        augment=False,
        feature_engineering=feature_engineering,
    )
    candidates = [int(item) for item in args.batch_candidates.split(",") if item.strip()]
    batch_size = args.batch_size or autotune_batch(train_ds, candidates, args, device)
    print(f"selected batch_size: {batch_size}")
    train_loader, train_sampler = make_loader(
        train_ds, batch_size, args.num_workers, args.prefetch_factor, True, args.seed, device.type == "cuda"
    )
    test_loader, _ = make_loader(
        test_ds, batch_size, args.num_workers, args.prefetch_factor, False, args.seed, device.type == "cuda"
    )
    print(f"images train={len(train_ds)} test={len(test_ds)} bagsize={int(train_ds.df.iloc[0]['count'])}")

    model = VNNMILAttention(in_ch=in_ch).to(device)
    train_labels = train_ds.df["y"].astype(int)
    pos = max(1, int(train_labels.sum()))
    neg = max(1, int((train_labels == 0).sum()))
    pos_weight = torch.tensor([neg / pos], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, object]] = []
    best_pr_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        images_seen = 0
        patches_seen = 0
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for x, y, valid, _metadata in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            if device.type != "cuda":
                x = x.float()
            y = y.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            sample_weight = build_sample_weights(_metadata, len(y), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = model(x, valid)
                loss = weighted_loss_fn(
                    logits,
                    y,
                    sample_weight,
                    pos_weight,
                    args.loss_type,
                    args.focal_gamma,
                    args.label_smoothing,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(y)
            images_seen += len(y)
            patches_seen += int(valid.sum().item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        predictions, top_patches = evaluate(model, test_loader, device, args.attention_top_k)
        metrics = group_metrics(predictions)
        pr_auc = metrics["overall"]["pr_auc"] or 0.0
        epoch_row = {
            "epoch": epoch,
            "loss": total_loss / max(1, images_seen),
            "seconds": elapsed,
            "seconds_per_batch": elapsed / max(1, len(train_loader)),
            "patches_per_second": patches_seen / max(elapsed, 1e-9),
            **metrics["overall"],
        }
        if device.type == "cuda":
            epoch_row["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            epoch_row["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        history.append(epoch_row)
        print(epoch_row)
        if float(pr_auc) > best_pr_auc:
            best_pr_auc = float(pr_auc)
            torch.save({"model": model.state_dict(), "args": vars(args), "batch_size": batch_size}, out / "best_mil_vnn.pt")
            predictions.to_csv(out / "best_predictions.csv", index=False)
            top_patches.to_csv(out / "best_top_attention_patches.csv", index=False)
            (out / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions, top_patches = evaluate(model, test_loader, device, args.attention_top_k)
    final_metrics = group_metrics(predictions)
    run_info = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "amp": device.type == "cuda",
        "cudnn_benchmark": True,
        "manifest": args.manifest,
        "supervision": "MIL global image labels; attention is not segmentation ground truth",
        "metrics": final_metrics,
    }
    torch.save({"model": model.state_dict(), "args": vars(args), "batch_size": batch_size}, out / "last_mil_vnn.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    top_patches.to_csv(out / "top_attention_patches.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
