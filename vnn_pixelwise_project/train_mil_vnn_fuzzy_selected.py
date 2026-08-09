#!/usr/bin/env python3
"""Train VNN/Hybrid MIL on fuzzy-selected cached patch bags and compare to baseline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_hybrid_mil import json_default, load_manifest, metric_values, threshold_sweep, train_one


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MIL models using fuzzy chromatic selected patch shards.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--metadata-csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline-vnn-dir", help="Directory containing old VNN predictions/metrics.")
    p.add_argument("--mode", choices=["cnn", "vnn", "hybrid", "all"], default="all")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--aux-loss-weight", type=float, default=0.2)
    p.add_argument("--high-threshold", type=float, default=0.70)
    p.add_argument("--selection-priority", choices=["precision", "f1", "recall"], default="f1")
    p.add_argument("--feature-engineering", choices=["none", "stats"], default="stats")
    p.add_argument("--bag-drop-rate", type=float, default=0.10)
    p.add_argument("--channel-drop-rate", type=float, default=0.08)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--loss-type", choices=["bce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--selection-weight-floor", type=float, default=0.75)
    p.add_argument("--selection-weight-ceil", type=float, default=1.35)
    p.add_argument("--attention-top-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument(
        "--input-channels",
        choices=["all", "gray"],
        default="all",
        help="all usa 8 canais precomputados; gray usa somente o canal de intensidade.",
    )
    return p.parse_args()


def load_result(path: Path, label: str) -> dict[str, object] | None:
    predictions_path = path / "best_predictions.csv"
    if not predictions_path.exists():
        predictions_path = path / "predictions.csv"
    if not predictions_path.exists():
        return None
    predictions = pd.read_csv(predictions_path)
    y, score = predictions["y"].to_numpy(int), predictions["score"].to_numpy(float)
    _table, selected = threshold_sweep(y, score, 0.70, "f1")
    overall = metric_values(y, score, float(selected["threshold"]))
    return {"modelo": label, "ROC AUC": overall.get("roc_auc"), "PR AUC": overall.get("pr_auc"), "precision": overall.get("precision"), "recall": overall.get("recall"), "specificity": overall.get("specificity"), "F1": overall.get("f1"), "threshold": selected["threshold"]}


def train_original_vnn(args: argparse.Namespace) -> None:
    """Run the same VNNMILAttention implementation used by the old-selection baseline."""
    out = Path(args.out_dir) / "vnn"
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).with_name("train_mil_vnn_sharded.py")),
               "--manifest", args.manifest, "--out-dir", str(out), "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
               "--prefetch-factor", str(args.prefetch_factor), "--seed", str(args.seed),
               "--attention-top-k", str(args.attention_top_k), "--lr", str(args.lr),
               "--feature-engineering", args.feature_engineering, "--bag-drop-rate", str(args.bag_drop_rate),
               "--channel-drop-rate", str(args.channel_drop_rate), "--noise-std", str(args.noise_std),
               "--loss-type", args.loss_type, "--focal-gamma", str(args.focal_gamma),
               "--label-smoothing", str(args.label_smoothing), "--selection-weight-floor", str(args.selection_weight_floor),
               "--selection-weight-ceil", str(args.selection_weight_ceil)]
    if args.no_mmap:
        command.append("--no-mmap")
    print("Training VNN with baseline architecture: VNNMILAttention from train_mil_vnn_sharded.py")
    subprocess.run(command, check=True)
    source = out / "best_predictions.csv" if (out / "best_predictions.csv").exists() else out / "predictions.csv"
    predictions = pd.read_csv(source)
    y, score = predictions["y"].to_numpy(int), predictions["score"].to_numpy(float)
    sweep, selected = threshold_sweep(y, score, args.high_threshold, args.selection_priority)
    sweep.to_csv(out / "threshold_analysis.csv", index=False)
    (out / "best_threshold_sweep.json").write_text(json.dumps(selected, indent=2, default=json_default), encoding="utf-8")


def main() -> None:
    args = parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device); print("gpu:", torch.cuda.get_device_name(0) if device.type == "cuda" else "none")
    manifest = load_manifest(args)
    modes = ["vnn", "cnn", "hybrid"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print(f"\n=== Fuzzy-selected training: {mode} ===")
        if mode == "vnn":
            train_original_vnn(args)
        else:
            train_one(mode, manifest, args, device)
    rows = []
    if args.baseline_vnn_dir:
        baseline = load_result(Path(args.baseline_vnn_dir), "VNN selecao antiga")
        if baseline:
            rows.append(baseline)
    for mode, label in [("vnn", "VNN selecao fuzzy"), ("cnn", "CNN selecao fuzzy"), ("hybrid", "CNN+VNN selecao fuzzy")]:
        result = load_result(Path(args.out_dir) / mode, label)
        if result:
            rows.append(result)
    if rows:
        comparison = pd.DataFrame(rows).sort_values("PR AUC", ascending=False)
        comparison.to_csv(Path(args.out_dir) / "comparacao_selecao_fuzzy.csv", index=False)
        print("\n=== Comparison ==="); print(comparison.to_string(index=False))
    run = {"manifest": args.manifest, "baseline_vnn_dir": args.baseline_vnn_dir, "modes": modes, "epochs": args.epochs, "batch_size": args.batch_size, "selection_priority": args.selection_priority, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "vnn_architecture": "VNNMILAttention from train_mil_vnn_sharded.py, matching old-selection baseline", "selection_note": "Fuzzy selection uses cached patch features only; cancer label is not used for ranking.", "attention_note": "Attention is not segmentation ground truth."}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "fuzzy_selected_training_config.json").write_text(json.dumps(run, indent=2, default=json_default), encoding="utf-8")


if __name__ == "__main__":
    main()
