#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from fuzzy_cancer_rules import evaluate_fuzzy, save_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fuzzy cancer-tendency inference and evaluation from precomputed CSV features.")
    p.add_argument("--features_csv", required=True)
    p.add_argument("--predictions_csv", help="Optional model predictions merged when scores are absent in features CSV.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--parameters_json")
    return p.parse_args()


def json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def evaluate_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (score >= threshold).astype(int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    multi = len(np.unique(y)) > 1
    return {"n": int(len(y)), "positive": int(y.sum()), "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y, score)) if multi else None,
            "pr_auc": float(average_precision_score(y, score)) if multi else None,
            "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "sensitivity": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1)), "f1": float(f1_score(y, pred, zero_division=0)),
            "confusion_matrix": matrix.tolist(), "youden_index": float(tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1.0)}


def threshold_analysis(y: np.ndarray, score: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    table = pd.DataFrame([evaluate_metrics(y, score, value) for value in np.round(np.arange(0.01, 1.0, 0.01), 2)])
    best_f1 = table.sort_values(["f1", "recall", "specificity"], ascending=False).iloc[0].to_dict()
    best_youden = table.sort_values(["youden_index", "f1"], ascending=False).iloc[0].to_dict()
    def recall_target(target: float):
        valid = table[table.recall >= target]
        return None if valid.empty else valid.sort_values(["specificity", "precision", "threshold"], ascending=False).iloc[0].to_dict()
    return table, {"best_f1": best_f1, "recall_at_least_080": recall_target(0.80),
                   "recall_at_least_090": recall_target(0.90), "best_youden": best_youden,
                   "warning": "Threshold selection on evaluation data is exploratory and must not be reported as independent validation."}


def group_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    def one(part: pd.DataFrame):
        if part.empty: return None
        return evaluate_metrics(part["y"].to_numpy(int), part["tendencia_cancer"].to_numpy(float), threshold)
    result = {"overall": one(frame)}
    result["by_view"] = {str(key): one(group) for key, group in frame.groupby("view", dropna=False)} if "view" in frame else {}
    result["by_density"] = {str(key): one(group) for key, group in frame.groupby("density", dropna=False)} if "density" in frame else {}
    if "difficult_negative_case" in frame:
        difficult = frame["difficult_negative_case"].map(lambda x: str(x).lower() in {"true", "1", "t", "yes"})
        result["difficult_negative_case"] = {"included": one(frame), "excluded": one(frame[~difficult]), "only_difficult": one(frame[difficult])}
    return result


def load_inputs(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.features_csv, dtype={"patient_id": str, "image_id": str})
    parameters = json.loads(Path(args.parameters_json).read_text(encoding="utf-8")) if args.parameters_json else None
    if args.predictions_csv:
        pred = pd.read_csv(args.predictions_csv, dtype={"patient_id": str, "image_id": str})
        additions = [column for column in ["score", "score_vnn", "score_cnn", "score_hybrid", "y", "view", "density", "difficult_negative_case"] if column in pred and column not in frame]
        if additions:
            frame = frame.merge(pred[["patient_id", "image_id", *additions]], on=["patient_id", "image_id"], how="left")
    if "y" not in frame and "cancer" in frame:
        frame["y"] = frame["cancer"]
    if "y" not in frame:
        raise ValueError("Features/predictions must contain `y` or `cancer` for metrics.")
    if "tendencia_cancer" not in frame or parameters is not None:
        outcomes = frame.apply(lambda row: evaluate_fuzzy(row.to_dict(), parameters), axis=1)
        frame["tendencia_cancer"] = [out["tendencia_cancer"] for out in outcomes]
        frame["classe_fuzzy"] = [out["classe_fuzzy"] for out in outcomes]
        frame["regras_ativas"] = [json.dumps(out["regras_ativas"], ensure_ascii=True) for out in outcomes]
        frame["explicacao"] = [out["explicacao"] for out in outcomes]
    return frame


def main() -> None:
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frame = load_inputs(args)
    y, score = frame["y"].to_numpy(int), frame["tendencia_cancer"].to_numpy(float)
    sweep, selected = threshold_analysis(y, score)
    threshold = float(selected["best_f1"]["threshold"])
    frame["fuzzy_pred"] = (frame["tendencia_cancer"] >= threshold).astype(int)
    metrics = group_metrics(frame, threshold)
    frame.to_csv(out / "fuzzy_predictions.csv", index=False)
    sweep.to_csv(out / "fuzzy_threshold_analysis.csv", index=False)
    (out / "fuzzy_metrics.json").write_text(json.dumps({"selected_thresholds": selected, "metrics": metrics["overall"]}, indent=2, default=json_default), encoding="utf-8")
    (out / "fuzzy_metrics_by_group.json").write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")
    save_parameters(out / "fuzzy_parameters.json")
    print(json.dumps({"threshold": threshold, **metrics["overall"]}, indent=2, default=json_default))
    print("Saida:", out)


if __name__ == "__main__":
    main()
