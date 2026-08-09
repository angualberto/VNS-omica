from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hybrid_cnn_vnn_mil import HybridCNNVNNMIL
from sharded_dataset import ShardBatchSampler, ShardedMILDataset


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def normalize_bool_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "t", "yes", "y"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN, VNN and CNN+VNN MIL models from existing shards.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--metadata-csv", help="Original RSNA train.csv, used to restore laterality and verify metadata.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["cnn", "vnn", "hybrid", "all"], default="all")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--aux-loss-weight", type=float, default=0.2, help="Hybrid-only supervision for branch scores.")
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--selection-priority", choices=["precision", "f1", "recall"], default="f1")
    p.add_argument("--feature-engineering", choices=["none", "stats"], default="stats",
                   help="stats appends per-patch mean/std channels computed from the cached feature stack.")
    p.add_argument("--bag-drop-rate", type=float, default=0.10, help="Randomly disable some valid patches during training.")
    p.add_argument("--channel-drop-rate", type=float, default=0.08, help="Randomly zero feature channels during training.")
    p.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise added to training bags after loading.")
    p.add_argument("--loss-type", choices=["bce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--selection-weight-floor", type=float, default=0.75)
    p.add_argument("--selection-weight-ceil", type=float, default=1.35)
    p.add_argument("--attention-top-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--max-images", type=int, default=0, help="Development-only cap per split.")
    p.add_argument("--input-channels", choices=["all", "gray"], default="all",
                   help="all usa 8 canais precomputados; gray usa somente a imagem bruta, sem filtros espectrais.")
    return p.parse_args()


def load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest)
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest["image_id"] = manifest["image_id"].astype(str)
    if args.metadata_csv:
        meta = pd.read_csv(args.metadata_csv, dtype={"patient_id": str, "image_id": str})
        keep = ["patient_id", "image_id", "laterality", "view", "density", "difficult_negative_case"]
        keep = [col for col in keep if col in meta.columns]
        merged = manifest.merge(meta[keep].drop_duplicates(["patient_id", "image_id"]),
                                on=["patient_id", "image_id"], how="left", suffixes=("", "_rsna"))
        for col in ["laterality", "view", "density", "difficult_negative_case"]:
            source = f"{col}_rsna"
            if source in merged.columns:
                merged[col] = merged[source].combine_first(merged[col]) if col in merged.columns else merged[source]
                merged = merged.drop(columns=[source])
        manifest = merged
    for col, default in {"laterality": "", "view": "UNKNOWN", "density": "UNKNOWN", "difficult_negative_case": False}.items():
        if col not in manifest.columns:
            manifest[col] = default
        manifest[col] = manifest[col].fillna(default)
    manifest["difficult_negative_case"] = normalize_bool_column(manifest["difficult_negative_case"])
    if "split" not in manifest or not {"train", "test"}.issubset(set(manifest["split"].astype(str))):
        raise ValueError("Manifest must contain train/test splits created before preprocessing.")
    train_patients = set(manifest.loc[manifest.split == "train", "patient_id"])
    test_patients = set(manifest.loc[manifest.split == "test", "patient_id"])
    if train_patients & test_patients:
        raise ValueError("Patient leakage: one or more patient_id values occur in train and test.")
    if args.max_images > 0:
        parts = [group.head(args.max_images) for _split, group in manifest.groupby("split", sort=False)]
        manifest = pd.concat(parts, ignore_index=True)
    return manifest


def make_loader(dataset: ShardedMILDataset, args: argparse.Namespace, training: bool, cuda: bool):
    sampler = ShardBatchSampler(dataset, args.batch_size, shuffle=training, seed=args.seed)
    kwargs = {"batch_sampler": sampler, "num_workers": args.num_workers, "pin_memory": cuda,
              "persistent_workers": args.num_workers > 0}
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs), sampler


def metric_values(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (score >= threshold).astype(int)
    multi = len(np.unique(y)) > 1
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, _fn, _tp = matrix.ravel()
    return {
        "n": int(len(y)), "positive": int(y.sum()), "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, score)) if multi else None,
        "pr_auc": float(average_precision_score(y, score)) if multi else None,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix": matrix.tolist(),
    }


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
    weights = [float(metadata_item(metadata, "selection_weight", i)) if "selection_weight" in metadata else 1.0 for i in range(batch_size)]
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
    base = torch.nn.functional.binary_cross_entropy_with_logits(
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


def threshold_sweep(y: np.ndarray, score: np.ndarray, high_threshold: float, priority: str = "f1") -> tuple[pd.DataFrame, dict[str, object]]:
    rows = [metric_values(y, score, threshold) for threshold in np.round(np.arange(0.01, 0.901, 0.01), 2)]
    table = pd.DataFrame(rows)
    if priority == "precision":
        best = table.sort_values(["precision", "recall", "f1"], ascending=False).iloc[0].to_dict()
    elif priority == "recall":
        best = table.sort_values(["recall", "precision", "f1"], ascending=False).iloc[0].to_dict()
    else:
        best = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0].to_dict()
    best["fixed_high_threshold"] = metric_values(y, score, high_threshold)
    return table, best


def subgroup_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, object]:
    def one(frame: pd.DataFrame):
        return metric_values(frame.y.to_numpy(int), frame.score.to_numpy(float), threshold)
    result: dict[str, object] = {"overall": one(predictions)}
    result["by_view"] = {str(k): one(v) for k, v in predictions.groupby("view", dropna=False)}
    result["by_density"] = {str(k): one(v) for k, v in predictions.groupby("density", dropna=False)}
    result["by_difficult_negative_case"] = {str(k): one(v) for k, v in predictions.groupby("difficult_negative_case")}
    result["excluding_difficult_negative_case"] = one(predictions[~predictions.difficult_negative_case.astype(bool)])
    return result


def metadata_item(metadata: dict, name: str, index: int):
    item = metadata[name][index]
    return item.item() if torch.is_tensor(item) else item


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, attention_top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    predictions, attention_rows = [], []
    with torch.inference_mode():
        for x, y, valid, metadata in loader:
            x = x.to(device, non_blocking=True)
            if device.type != "cuda":
                x = x.float()
            valid_gpu = valid.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, details = model(x, valid_gpu, return_details=True)
            attention = details["attention"].float().cpu()
            score = torch.sigmoid(logits).float().cpu().numpy()
            score_cnn = torch.sigmoid(details["cnn_logits"]).float().cpu().numpy() if details["cnn_logits"] is not None else None
            score_vnn = torch.sigmoid(details["vnn_logits"]).float().cpu().numpy() if details["vnn_logits"] is not None else None
            for i in range(len(y)):
                row = {"patient_id": str(metadata_item(metadata, "patient_id", i)),
                       "image_id": str(metadata_item(metadata, "image_id", i)),
                       "image_path": str(metadata_item(metadata, "image_path", i)),
                       "laterality": str(metadata_item(metadata, "laterality", i)),
                       "view": str(metadata_item(metadata, "view", i)),
                       "density": str(metadata_item(metadata, "density", i)),
                       "difficult_negative_case": bool(metadata_item(metadata, "difficult_negative_case", i)),
                       "y": int(y[i].item()), "score": float(score[i])}
                if score_cnn is not None:
                    row.update({"score_cnn": float(score_cnn[i]), "score_vnn": float(score_vnn[i])})
                predictions.append(row)
                valid_index = torch.where(valid[i])[0]
                ranking = valid_index[torch.argsort(attention[i, valid_index], descending=True)]
                for rank, patch_id in enumerate(ranking.tolist()[:attention_top_k], start=1):
                    coords = metadata["coords"][i, patch_id]
                    attention_rows.append({**row, "rank": rank, "patch_index": patch_id,
                                           "attention": float(attention[i, patch_id]),
                                           "x0": int(coords[0]), "y0": int(coords[1])})
    return pd.DataFrame(predictions), pd.DataFrame(attention_rows)


def save_curves(predictions: pd.DataFrame, out: Path) -> None:
    y, score = predictions.y.to_numpy(int), predictions.score.to_numpy(float)
    if len(np.unique(y)) < 2:
        return
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC")
    plt.tight_layout(); plt.savefig(out / "roc.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 5)); plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.tight_layout(); plt.savefig(out / "pr.png", dpi=160); plt.close()


def train_one(mode: str, manifest: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> None:
    out = Path(args.out_dir) / mode
    out.mkdir(parents=True, exist_ok=True)
    channels = (0,) if args.input_channels == "gray" else None
    feature_engineering = args.feature_engineering if channels is None else "none"
    in_ch = 1 if channels is not None else 8
    if feature_engineering == "stats":
        in_ch += 2
    spectral_filters = args.input_channels != "gray"
    print(f"input_channels={args.input_channels}; feature_engineering={feature_engineering}; in_ch={in_ch}; spectral_filters={spectral_filters}")
    train_ds = ShardedMILDataset(
        manifest,
        split="train",
        mmap=not args.no_mmap,
        channels=channels,
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
        channels=channels,
        training=False,
        augment=False,
        feature_engineering=feature_engineering,
    )
    train_loader, sampler = make_loader(train_ds, args, True, device.type == "cuda")
    test_loader, _ = make_loader(test_ds, args, False, device.type == "cuda")
    model = HybridCNNVNNMIL(mode=mode, in_ch=in_ch).to(device)
    labels = train_ds.df.y.astype(int)
    pos, neg = max(1, int(labels.sum())), max(1, int((labels == 0).sum()))
    pos_weight = torch.tensor([neg / pos], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_key = [], (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch); model.train(); total_loss = 0.0; bags = 0; patches = 0
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for x, y, valid, _meta in tqdm(train_loader, desc=f"{mode} epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True); valid = valid.to(device, non_blocking=True)
            if device.type != "cuda": x = x.float()
            sample_weight = build_sample_weights(_meta, len(y), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, details = model(x, valid, return_details=True)
                loss = weighted_loss_fn(
                    logits,
                    y,
                    sample_weight,
                    pos_weight,
                    args.loss_type,
                    args.focal_gamma,
                    args.label_smoothing,
                )
                if mode == "hybrid" and args.aux_loss_weight > 0:
                    loss = loss + args.aux_loss_weight * (
                        weighted_loss_fn(
                            details["cnn_logits"],
                            y,
                            sample_weight,
                            pos_weight,
                            args.loss_type,
                            args.focal_gamma,
                            args.label_smoothing,
                        )
                        + weighted_loss_fn(
                            details["vnn_logits"],
                            y,
                            sample_weight,
                            pos_weight,
                            args.loss_type,
                            args.focal_gamma,
                            args.label_smoothing,
                        )
                    )
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += float(loss.item()) * len(y); bags += len(y); patches += int(valid.sum().item())
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        predictions, top_attention = evaluate(model, test_loader, device, args.attention_top_k)
        threshold_table, best_threshold = threshold_sweep(
            predictions.y.to_numpy(int),
            predictions.score.to_numpy(float),
            args.high_threshold,
            args.selection_priority,
        )
        metrics = subgroup_metrics(predictions, float(best_threshold["threshold"]))
        overall = metrics["overall"]
        row = {"model": mode, "epoch": epoch, "loss": total_loss / max(bags, 1), "seconds": elapsed,
               "seconds_per_batch": elapsed / max(len(train_loader), 1), "bags_per_second": bags / max(elapsed, 1e-9),
               "patches_per_second": patches / max(elapsed, 1e-9), **overall,
               "recall_threshold_high": best_threshold["fixed_high_threshold"]["recall"]}
        if device.type == "cuda": row["peak_vram_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        history.append(row); print(row)
        key = (float(overall["pr_auc"] or 0.0), float(overall["roc_auc"] or 0.0))
        if key > best_key:
            best_key = key
            torch.save({"model": model.state_dict(), "mode": mode, "args": vars(args), "epoch": epoch,
                        "metrics": metrics, "best_threshold": best_threshold}, out / "best_model.pt")
            predictions.assign(pred=(predictions.score >= float(best_threshold["threshold"])).astype(int)).to_csv(out / "predictions.csv", index=False)
            top_attention.to_csv(out / "top_attention_patches.csv", index=False)
            threshold_table.to_csv(out / "threshold_analysis.csv", index=False)
            (out / "metrics.json").write_text(
                json.dumps({"model": mode, "metrics": metrics, "best_threshold": best_threshold}, indent=2, default=json_default),
                encoding="utf-8",
            )
            save_curves(predictions, out)
        pd.DataFrame(history).to_csv(out / "metrics_epoch.csv", index=False)


def main() -> None:
    args = parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device); print("gpu:", torch.cuda.get_device_name(0) if device.type == "cuda" else "none")
    manifest = attach_selection_weights(load_manifest(args), args.manifest, args.selection_weight_floor, args.selection_weight_ceil)
    modes = ["cnn", "vnn", "hybrid"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print(f"\n=== Training {mode} ===")
        train_one(mode, manifest, args, device)


if __name__ == "__main__":
    main()
