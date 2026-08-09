#!/usr/bin/env python3
"""Converte o CSV `cores_alizams_por_imagem*.csv` em um dicionário JSON.

Exemplo:
  python convert_colors_to_dict.py \
    --csv /path/cores_alizams_por_imagem_all_dedup.csv \
    --out /path/cores_dict.json --key image_id
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import pandas as pd


def numpy_safe(x: Any):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def csv_to_dict(csv_path: str, key: str) -> dict:
    df = pd.read_csv(csv_path)
    if key not in df.columns:
        raise KeyError(f"Chave '{key}' não encontrada no CSV. Colunas: {list(df.columns)}")
    df = df.fillna(value=np.nan)
    out = {}
    for _, row in df.iterrows():
        k = row[key]
        if pd.isna(k):
            continue
        # map all other columns
        rowd = {c: numpy_safe(row[c]) for c in df.columns if c != key}
        out[str(k)] = rowd
    return out


def main():
    p = argparse.ArgumentParser(description="Converter CSV de cores -> dicionário JSON")
    p.add_argument("--csv", required=True, help="CSV de entrada (cores_alizams_por_imagem_*.csv)")
    p.add_argument("--out", required=True, help="Caminho do JSON de saída")
    p.add_argument("--key", default="image_id", help="Coluna a usar como chave (image_id ou image_path)")
    args = p.parse_args()

    try:
        d = csv_to_dict(args.csv, args.key)
    except Exception as e:
        print(f"Erro ao converter CSV: {e}", file=sys.stderr)
        sys.exit(2)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    print(f"Salvo {len(d)} registros em {args.out}")


if __name__ == "__main__":
    main()
