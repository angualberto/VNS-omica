#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_fuzzy_inference import evaluate_metrics, threshold_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CNN/VNN/Hybrid and fuzzy tendency scores.")
    p.add_argument("--models_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def locate(directory: Path) -> Path | None:
    for name in ["fuzzy_predictions.csv", "predictions.csv", "best_predictions.csv"]:
        if (directory / name).exists(): return directory / name
    return None


def model_name(path: Path, index: int) -> str:
    text = str(path).lower()
    if "fuzzy" in text and "hybrid" in text: return "Fuzzy sobre Hybrid"
    if "fuzzy" in text: return "Fuzzy sobre VNN"
    if "hybrid" in text: return "Hybrid CNN+VNN"
    if "cnn" in text: return "CNN"
    if "vnn" in text: return "VNN"
    return f"Modelo {index + 1}"


def load_score(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str})
    if "y" not in frame and "cancer" in frame: frame["y"] = frame["cancer"]
    score = "tendencia_cancer" if "tendencia_cancer" in frame else "score"
    if score not in frame: raise ValueError(f"No score in {path}")
    return frame[["patient_id", "image_id", "y", score]].rename(columns={score: label})


def main() -> None:
    args = parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    models, rows = {}, []
    for index, directory in enumerate(map(Path, args.models_dirs)):
        source = locate(directory)
        if source is None: continue
        label = model_name(directory, index)
        models[label] = load_score(source, label)
    for label, frame in models.items():
        sweep, chosen = threshold_analysis(frame.y.to_numpy(int), frame[label].to_numpy(float))
        metric = evaluate_metrics(frame.y.to_numpy(int), frame[label].to_numpy(float), float(chosen["best_f1"]["threshold"]))
        rows.append({"modelo": label, "ROC AUC": metric["roc_auc"], "PR AUC": metric["pr_auc"], "precision": metric["precision"],
                     "recall": metric["recall"], "specificity": metric["specificity"], "F1": metric["f1"], "threshold": metric["threshold"]})
    base_models = [key for key in models if not key.startswith("Fuzzy")]
    if len(base_models) >= 2:
        joined = models[base_models[0]].copy()
        for label in base_models[1:]: joined = joined.merge(models[label].drop(columns="y"), on=["patient_id", "image_id"], how="inner")
        joined["Ensemble score medio"] = joined[base_models].mean(axis=1)
        metric_table, chosen = threshold_analysis(joined.y.to_numpy(int), joined["Ensemble score medio"].to_numpy(float))
        metric = evaluate_metrics(joined.y.to_numpy(int), joined["Ensemble score medio"].to_numpy(float), float(chosen["best_f1"]["threshold"]))
        rows.append({"modelo": "Ensemble score medio", "ROC AUC": metric["roc_auc"], "PR AUC": metric["pr_auc"], "precision": metric["precision"], "recall": metric["recall"], "specificity": metric["specificity"], "F1": metric["f1"], "threshold": metric["threshold"]})
    fuzzy_models = [key for key in models if key.startswith("Fuzzy")]
    if fuzzy_models and base_models:
        fuzzy, base = fuzzy_models[0], base_models[-1]
        joined = models[fuzzy].merge(models[base].drop(columns="y"), on=["patient_id", "image_id"], how="inner")
        joined["Ensemble fuzzy final"] = (joined[fuzzy] + joined[base]) / 2.0
        _, chosen = threshold_analysis(joined.y.to_numpy(int), joined["Ensemble fuzzy final"].to_numpy(float))
        metric = evaluate_metrics(joined.y.to_numpy(int), joined["Ensemble fuzzy final"].to_numpy(float), float(chosen["best_f1"]["threshold"]))
        rows.append({"modelo": "Ensemble fuzzy final", "ROC AUC": metric["roc_auc"], "PR AUC": metric["pr_auc"], "precision": metric["precision"], "recall": metric["recall"], "specificity": metric["specificity"], "F1": metric["f1"], "threshold": metric["threshold"]})
    table = pd.DataFrame(rows).sort_values("PR AUC", ascending=False)
    table.to_csv(output / "comparacao_fuzzy_vs_modelos.csv", index=False)
    print(table.to_string(index=False)); print("Saida:", output)


if __name__ == "__main__":
    main()
