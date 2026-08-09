#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot fuzzy tendency outputs from generated CSV only.")
    p.add_argument("--fuzzy_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--threshold", type=float)
    return p.parse_args()


def main() -> None:
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.fuzzy_csv)
    if "y" not in frame and "cancer" in frame: frame["y"] = frame["cancer"]
    y, score = frame.y.astype(int), frame.tendencia_cancer.astype(float)
    threshold = args.threshold if args.threshold is not None else 0.5
    for label, group in frame.groupby("y"):
        plt.hist(group["tendencia_cancer"], bins=35, density=True, histtype="step", label=f"y={label}")
    plt.legend()
    plt.title("Tendencia fuzzy por classe real"); plt.tight_layout(); plt.savefig(out / "histograma_tendencia_por_classe.png", dpi=170); plt.close()
    fpr, tpr, _ = roc_curve(y, score); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC fuzzy"); plt.tight_layout(); plt.savefig(out / "roc_fuzzy.png", dpi=170); plt.close()
    precision, recall, _ = precision_recall_curve(y, score); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR fuzzy"); plt.tight_layout(); plt.savefig(out / "pr_fuzzy.png", dpi=170); plt.close()
    for feature in ["magenta_ratio", "hue_medio"]:
        if feature in frame:
            for label, group in frame.groupby("y"):
                plt.hist(group[feature].dropna(), bins=35, density=True, histtype="step", label=f"y={label}")
            plt.legend()
            plt.title(f"Distribuicao de {feature} por classe"); plt.tight_layout(); plt.savefig(out / f"distribuicao_{feature}.png", dpi=170); plt.close()
    for feature in ["density", "view"]:
        if feature in frame:
            plt.figure(figsize=(9, 5)); frame.boxplot(column="tendencia_cancer", by=feature, grid=False)
            plt.suptitle("")
            plt.title(f"Tendencia fuzzy por {feature}"); plt.tight_layout(); plt.savefig(out / f"boxplot_por_{feature}.png", dpi=170); plt.close()
    counts = Counter()
    if "regras_ativas" in frame:
        for raw in frame.regras_ativas.dropna():
            for rule in json.loads(raw): counts[rule["regra"]] += float(rule["intensidade"])
    if counts:
        rule_frame = pd.DataFrame(counts.items(), columns=["regra", "ativacao_total"]).sort_values("ativacao_total", ascending=False)
        plt.bar(rule_frame["regra"], rule_frame["ativacao_total"]); plt.title("Regras fuzzy mais ativadas"); plt.tight_layout(); plt.savefig(out / "regras_mais_ativadas.png", dpi=170); plt.close()
        rule_frame.to_csv(out / "ativacao_regras.csv", index=False)
    pred = (score >= threshold).astype(int); ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1])).plot(cmap="Blues")
    plt.title(f"Matriz de confusao fuzzy (threshold={threshold:.2f})"); plt.tight_layout(); plt.savefig(out / "matriz_confusao_fuzzy.png", dpi=170); plt.close()
    print("Saida:", out)


if __name__ == "__main__":
    main()
