#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuzzy_cancer_rules import evaluate_fuzzy, save_parameters


KEYS = ["patient_id", "image_id"]
STRUCTURAL_MAP = {
    "grad_mean_mean": "gradiente_medio", "grad_top20_mean_max": "gradiente_top",
    "entropy_hist_mean": "entropia_local", "wave_high_prop_mean": "wavelet_high",
    "fft_high_mean": "fft_high", "fft_high_low_ratio_mean": "high_low_ratio",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build image-level fuzzy input table from generated CSV outputs only.")
    p.add_argument("--predictions-csv", required=True)
    p.add_argument("--colors-csv", required=True, help="Precomputed AlizaMS/RainbowB feature CSV, Python or CUDA Fortran output.")
    p.add_argument("--attention-csv")
    p.add_argument("--structural-csv", help="Optional precomputed gradient/entropy/wavelet/FFT feature CSV.")
    p.add_argument("--model-mode", choices=["vnn", "cnn", "hybrid"], default="vnn")
    p.add_argument("--vnn-predictions-csv", help="Optional VNN predictions to add score_vnn.")
    p.add_argument("--cnn-predictions-csv", help="Optional CNN predictions to add score_cnn.")
    p.add_argument("--hybrid-predictions-csv", help="Optional hybrid predictions to add score_hybrid.")
    p.add_argument("--bag-size", type=int, default=8)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--parameters-json")
    return p.parse_args()


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    for key in KEYS:
        frame[key] = frame[key].astype(str)
    return frame


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "t", "yes", "y"})


def attention_summary(path: str, bag_size: int) -> pd.DataFrame:
    attention = normalize_keys(pd.read_csv(path, dtype={"patient_id": str, "image_id": str}))
    rows = []
    for keys, group in attention.groupby(KEYS, sort=False):
        values = group["attention"].astype(float).clip(lower=0).to_numpy()
        unknown = max(0, bag_size - len(values))
        remainder = max(0.0, 1.0 - values.sum())
        distribution = np.r_[values, np.repeat(remainder / unknown, unknown)] if unknown else values
        distribution = distribution / max(distribution.sum(), 1e-12)
        entropy = -float(np.sum(distribution * np.log(distribution + 1e-12))) / np.log(max(bag_size, 2))
        rows.append({"patient_id": keys[0], "image_id": keys[1], "attention_max": float(values.max()),
                     "attention_entropy": entropy, "attention_concentration": float(values.max()),
                     "attention_is_approximated_from_top_patches": len(values) < bag_size})
    return pd.DataFrame(rows)


def merge_model_score(table: pd.DataFrame, path: str | None, mode: str) -> pd.DataFrame:
    if not path:
        return table
    score_name = f"score_{mode}"
    extra = normalize_keys(pd.read_csv(path, dtype={"patient_id": str, "image_id": str}))
    if score_name not in extra.columns and "score" in extra.columns:
        extra[score_name] = extra["score"]
    if score_name not in extra.columns:
        raise ValueError(f"Prediction CSV for {mode} lacks `score` or `{score_name}`: {path}")
    values = extra[KEYS + [score_name]].drop_duplicates(KEYS)
    if score_name in table.columns:
        table = table.drop(columns=score_name)
    return table.merge(values, on=KEYS, how="left", validate="one_to_one")


def build_features(args: argparse.Namespace) -> pd.DataFrame:
    pred = normalize_keys(pd.read_csv(args.predictions_csv, dtype={"patient_id": str, "image_id": str}))
    if "y" not in pred.columns and "cancer" in pred.columns:
        pred["y"] = pred["cancer"]
    if "cancer" not in pred.columns and "y" in pred.columns:
        pred["cancer"] = pred["y"]
    score_name = f"score_{args.model_mode}"
    if score_name not in pred.columns and "score" in pred.columns:
        pred[score_name] = pred["score"]
    colors = normalize_keys(pd.read_csv(args.colors_csv, dtype={"patient_id": str, "image_id": str}))
    colors = colors.rename(columns={"hue_medio_graus": "hue_medio"})
    color_columns = KEYS + [col for col in ["hue_medio", "freq_roxo", "freq_rosa", "freq_azul", "freq_verde", "freq_ciano", "lut", "laterality"] if col in colors.columns and col not in pred.columns]
    table = pred.merge(colors[color_columns].drop_duplicates(KEYS), on=KEYS, how="left", validate="one_to_one")
    for mode in ["vnn", "cnn", "hybrid"]:
        table = merge_model_score(table, getattr(args, f"{mode}_predictions_csv"), mode)
    if args.attention_csv:
        table = table.merge(attention_summary(args.attention_csv, args.bag_size), on=KEYS, how="left", validate="one_to_one")
    if args.structural_csv:
        structural = normalize_keys(pd.read_csv(args.structural_csv, dtype={"patient_id": str, "image_id": str}))
        cols = KEYS + [col for col in STRUCTURAL_MAP if col in structural.columns]
        table = table.merge(structural[cols].drop_duplicates(KEYS).rename(columns=STRUCTURAL_MAP), on=KEYS, how="left")
    for name in ["score_vnn", "score_cnn", "score_hybrid", "attention_max", "attention_entropy", "attention_concentration",
                 "hue_medio", "freq_roxo", "freq_rosa", "freq_azul", "freq_verde", "freq_ciano",
                 "gradiente_medio", "gradiente_top", "entropia_local", "wavelet_high", "fft_high", "high_low_ratio"]:
        if name not in table.columns:
            table[name] = np.nan
    if "difficult_negative_case" in table.columns:
        table["difficult_negative_case"] = bool_series(table["difficult_negative_case"])
    eps = 1e-8
    table["magenta_ratio"] = (table["freq_roxo"] + table["freq_rosa"]) / (table["freq_azul"] + table["freq_verde"] + eps)
    table["roxo_menos_azul"] = table["freq_roxo"] - table["freq_azul"]
    table["magenta_minus_cold"] = (table["freq_roxo"] + table["freq_rosa"]) - (table["freq_azul"] + table["freq_verde"] + table["freq_ciano"])
    results = table.apply(lambda row: evaluate_fuzzy(row.to_dict()), axis=1)
    table["tendencia_cancer"] = [item["tendencia_cancer"] for item in results]
    table["classe_fuzzy"] = [item["classe_fuzzy"] for item in results]
    table["regras_ativas"] = [json.dumps(item["regras_ativas"], ensure_ascii=True) for item in results]
    table["explicacao"] = [item["explicacao"] for item in results]
    return table


def main() -> None:
    args = parse_args()
    output = Path(args.output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    table = build_features(args)
    table.to_csv(output, index=False)
    parameters = Path(args.parameters_json) if args.parameters_json else output.parent / "fuzzy_parameters.json"
    save_parameters(parameters)
    source_text = str(args.colors_csv).lower()
    if "cuda" not in source_text and "fortran" not in source_text:
        print("Aviso: CSV cromatico carregado sem identificacao CUDA Fortran; usando features precomputadas existentes, sem recalcular DICOM.")
    print(f"imagens={len(table)} saida={output} parametros={parameters}")


if __name__ == "__main__":
    main()
