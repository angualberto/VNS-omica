#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FEATURES = ["score_vnn", "score_cnn", "score_hybrid", "attention_max", "attention_entropy", "attention_concentration",
            "hue_medio", "freq_roxo", "freq_rosa", "freq_azul", "freq_verde", "magenta_ratio", "magenta_minus_cold",
            "gradiente_top", "entropia_local", "wavelet_high", "fft_high"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain one fuzzy image-level statistical tendency result.")
    p.add_argument("--image_id", required=True)
    p.add_argument("--patient_id")
    p.add_argument("--fuzzy_csv", required=True)
    p.add_argument("--attention_dir", help="Directory or CSV containing stored top MIL attention patches/overlays.")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def attention_source(path: str | None) -> Path | None:
    if not path: return None
    source = Path(path)
    if source.is_file(): return source
    for candidate in ["top_attention_patches.csv", "best_top_attention_patches.csv", "attention_examples.csv"]:
        if (source / candidate).exists(): return source / candidate
    return None


def main() -> None:
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.fuzzy_csv, dtype={"patient_id": str, "image_id": str})
    selected = table[table.image_id == str(args.image_id)]
    if args.patient_id: selected = selected[selected.patient_id == str(args.patient_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one case, found {len(selected)}; provide --patient_id when image_id is ambiguous.")
    row = selected.iloc[0]
    rules = json.loads(row.get("regras_ativas", "[]"))
    values = {feature: None if pd.isna(row.get(feature)) else float(row[feature]) for feature in FEATURES if feature in row}
    case_id = f"{row.patient_id}_{row.image_id}"
    text = ["RELATORIO FUZZY - TENDENCIA ESTATISTICA, NAO DIAGNOSTICO MEDICO", "",
            f"patient_id: {row.patient_id}", f"image_id: {row.image_id}", f"view: {row.get('view', '')}",
            f"density: {row.get('density', '')}", f"tendencia_cancer: {float(row.tendencia_cancer):.4f}",
            f"classe_fuzzy: {row.classe_fuzzy}", "", "Explicacao:", str(row.get("explicacao", "")), "", "Regras ativadas:"]
    for rule in rules:
        text.append(f"- {rule['regra']} intensidade={rule['intensidade']:.4f}: {rule['descricao']}")
    text += ["", "Features:"] + [f"- {key}: {value}" for key, value in values.items()]
    attention_csv = attention_source(args.attention_dir)
    if attention_csv:
        attention = pd.read_csv(attention_csv, dtype={"patient_id": str, "image_id": str})
        patches = attention[(attention.patient_id == str(row.patient_id)) & (attention.image_id == str(row.image_id))].sort_values("attention", ascending=False)
        patches.to_csv(out / f"{case_id}_top_attention.csv", index=False)
        text += ["", "Top patches MIL (attention nao e ground truth de segmentacao):"]
        for patch in patches.head(8).itertuples(index=False):
            text.append(f"- patch={getattr(patch, 'patch_index', '')} attention={getattr(patch, 'attention', 0):.4f} coords=({getattr(patch, 'x0', '')},{getattr(patch, 'y0', '')})")
    if args.attention_dir and Path(args.attention_dir).is_dir():
        for image in Path(args.attention_dir).glob(f"*patient_{row.patient_id}_image_{row.image_id}*overlay*.png"):
            shutil.copy2(image, out / image.name)
    (out / f"{case_id}_explicacao.txt").write_text("\n".join(text), encoding="utf-8")
    displayed = {key: value for key, value in values.items() if value is not None}
    if displayed:
        plt.figure(figsize=(11, 4)); plt.bar(displayed.keys(), displayed.values()); plt.xticks(rotation=55, ha="right")
        plt.title(f"Features fuzzy - {case_id} | tendencia={row.tendencia_cancer:.3f}"); plt.tight_layout()
        plt.savefig(out / f"{case_id}_features.png", dpi=160); plt.close()
    print("\n".join(text)); print("Saida:", out)


if __name__ == "__main__":
    main()
