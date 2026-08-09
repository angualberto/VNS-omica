from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vnn_pixelwise_mammo import VNNPixelMammo


def load_cached_dataset_class():
    path = Path(__file__).with_name("2_dataset_cached.py")
    spec = importlib.util.spec_from_file_location("dataset_cached", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.CachedMammoPatchDataset


CachedMammoPatchDataset = load_cached_dataset_class()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VNN from cached patch tensors")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--top-k-pixels", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_images(manifest: pd.DataFrame, seed: int) -> tuple[set[str], set[str]]:
    images = manifest[["image_path", "cancer"]].drop_duplicates("image_path")
    if "split" in manifest.columns and set(manifest["split"].dropna().unique()).issuperset({"train", "test"}):
        train_paths = set(manifest.loc[manifest["split"] == "train", "image_path"])
        test_paths = set(manifest.loc[manifest["split"] == "test", "image_path"])
        return train_paths, test_paths
    train, test = train_test_split(images, test_size=0.25, stratify=images["cancer"], random_state=seed)
    return set(train["image_path"]), set(test["image_path"])


def global_logits(logits_map: torch.Tensor, top_k: int) -> torch.Tensor:
    flat = logits_map.flatten(1)
    k = min(max(1, top_k), flat.shape[1])
    return flat.topk(k, dim=1).values.mean(dim=1)


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    pred = (scores >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: str, top_k: int, output_csv: Path | None = None) -> dict[str, object]:
    model.eval()
    records = []
    with torch.no_grad():
        for x, y, image_code in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = global_logits(model(x), top_k)
            score = torch.sigmoid(logits).cpu().numpy()
            for yy, ss, cc in zip(y.numpy(), score, image_code.numpy()):
                records.append({"image_code": int(cc), "label": int(yy), "patch_score": float(ss)})
    patch_df = pd.DataFrame(records)
    grouped = patch_df.groupby("image_code", as_index=False).agg(label=("label", "first"), score=("patch_score", lambda s: float(np.mean(np.sort(s.to_numpy())[-min(8, len(s)):]))))
    grouped["pred"] = (grouped["score"] >= 0.5).astype(int)
    if output_csv is not None:
        grouped.to_csv(output_csv, index=False)
    return metrics_from_scores(grouped["label"].to_numpy(), grouped["score"].to_numpy())


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, cuda: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cuda,
        persistent_workers=(workers > 0),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    manifest = pd.read_csv(args.manifest)
    train_paths, test_paths = split_images(manifest, args.seed)
    train_ds = CachedMammoPatchDataset(manifest, train_paths, return_metadata=True)
    test_ds = CachedMammoPatchDataset(manifest, test_paths, return_metadata=True)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, True, device == "cuda")
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, False, device == "cuda")
    print(f"cached patches train={len(train_ds)} test={len(test_ds)}")

    model = VNNPixelMammo(in_ch=8).to(device)
    image_labels = manifest[manifest["image_path"].isin(train_paths)][["image_path", "cancer"]].drop_duplicates()["cancer"]
    pos = max(1, int(image_labels.sum()))
    neg = max(1, int((image_labels == 0).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    history = []
    best_pr = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total_loss = 0.0
        seen = 0
        for x, y, _image_code in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = global_logits(model(x), args.top_k_pixels)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        val = evaluate(model, test_loader, device, args.top_k_pixels)
        row = {"epoch": epoch, "loss": total_loss / max(seen, 1), **val}
        if device == "cuda":
            row["gpu_peak_allocated_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)
            row["gpu_peak_reserved_gb"] = float(torch.cuda.max_memory_reserved() / 1024**3)
        history.append(row)
        print(row)
        if val["pr_auc"] > best_pr:
            best_pr = float(val["pr_auc"])
            torch.save(model.state_dict(), out / "best_vnn_cached.pt")
    torch.save(model.state_dict(), out / "last_vnn_cached.pt")
    pd.DataFrame(history).to_csv(out / "historico_treino.csv", index=False)
    final_metrics = evaluate(model, test_loader, device, args.top_k_pixels, out / "predicoes.csv")
    final_metrics["train_patches"] = int(len(train_ds))
    final_metrics["test_patches"] = int(len(test_ds))
    final_metrics["device"] = device
    final_metrics["batch_size"] = int(args.batch_size)
    final_metrics["epochs"] = int(args.epochs)
    if device == "cuda":
        final_metrics["gpu_name"] = torch.cuda.get_device_name(0)
        final_metrics["gpu_reserved_gb_final"] = float(torch.cuda.memory_reserved() / 1024**3)
    (out / "metricas.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print("final:", json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
