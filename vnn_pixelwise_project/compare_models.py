from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CNN, VNN and hybrid MIL best results.")
    p.add_argument("--results-dir", required=True, help="Directory containing cnn/, vnn/ and hybrid/ outputs.")
    p.add_argument("--out-csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.results_dir)
    names = {"cnn": "CNN pura", "vnn": "VNN pura", "hybrid": "CNN+VNN hibrida"}
    rows = []
    for mode, label in names.items():
        path = base / mode / "metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        overall = payload["metrics"]["overall"]
        rows.append({"modelo": label, "mode": mode, "ROC AUC": overall.get("roc_auc"),
                     "PR AUC": overall.get("pr_auc"), "precision": overall.get("precision"),
                     "recall": overall.get("recall"), "specificity": overall.get("specificity"),
                     "F1": overall.get("f1"),
                     "melhor threshold": payload["best_threshold"].get("threshold")})
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {base}")
    table = pd.DataFrame(rows).sort_values("PR AUC", ascending=False)
    output = Path(args.out_csv) if args.out_csv else base / "comparacao_modelos.csv"
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print("saved:", output)


if __name__ == "__main__":
    main()
