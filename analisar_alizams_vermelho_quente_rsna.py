#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from lidc_sistema_integrado import read_rsna_image

ROOT = Path('/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem')
RUN_DIR = ROOT / 'sistema_integrado_rsna_csv_treino'
SPLIT_CSV = RUN_DIR / 'split_por_series_uid.csv'
ALIZAMS_LUTS = Path('/tmp/AlizaMS/common/luts.h')
OUT_DIR = RUN_DIR / 'alizams_rainbowb_vermelho_quente'
IMAGE_SIZE = 256
LUT_NAME = 'black_rainbow_lut'
MAX_COMPARISON_PER_CLASS = 6


def load_lut() -> np.ndarray:
    text = ALIZAMS_LUTS.read_text(encoding='utf-8', errors='ignore')
    m = re.search(rf'constexpr unsigned char {LUT_NAME}\[[^\]]+\]\s*=\s*\{{(.*?)\}};', text, re.S)
    nums = [int(x) for x in re.findall(r'\d+', m.group(1))]
    return np.asarray(nums, dtype=np.uint8).reshape(-1, 3)


def normalize(img: np.ndarray) -> np.ndarray:
    arr = cv2.resize(img.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    lo, hi = float(arr.min()), float(arr.max())
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def breast_mask(gray01: np.ndarray) -> np.ndarray:
    mask = gray01 > max(0.03, float(np.quantile(gray01, 0.08)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = labels == largest
    return clean if clean.mean() >= 0.01 else mask


def apply_lut(gray01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.floor(gray01 * len(lut)).astype(np.int32)
    idx = np.clip(idx, 0, len(lut) - 1)
    return lut[idx]


def red_hot_metrics(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    m = mask
    denom = float(m.sum() + 1e-6)

    # Vermelho visual forte: R alto, dominando G/B.
    red_dom = (r > 160) & (r > g * 1.15) & (r > b * 1.15) & m
    red_strong = (r > 220) & (r > g * 1.05) & (r > b * 1.05) & m
    red_score = np.clip((r - np.maximum(g, b)) / 255.0, 0, 1)
    red_score = red_score[m]

    return {
        'area_vermelho_dominante': float(red_dom.sum() / denom),
        'area_vermelho_forte': float(red_strong.sum() / denom),
        'score_vermelho_medio': float(red_score.mean()) if red_score.size else 0.0,
        'score_vermelho_p95': float(np.quantile(red_score, 0.95)) if red_score.size else 0.0,
        'canal_r_medio': float(r[m].mean()) if m.any() else 0.0,
        'canal_g_medio': float(g[m].mean()) if m.any() else 0.0,
        'canal_b_medio': float(b[m].mean()) if m.any() else 0.0,
    }


def cls(cancer: int) -> str:
    return 'cancer_verdadeiro' if int(cancer) else 'sem_cancer'


def analyze_row(row: pd.Series, lut: np.ndarray) -> dict[str, object]:
    img = read_rsna_image(Path(row['image_path']), 'mammography')
    gray = normalize(img)
    mask = breast_mask(gray)
    rgb = apply_lut(gray, lut)
    return {
        'split': row.get('split', ''),
        'patient_id': row.get('patient_id', ''),
        'image_id': row.get('image_id', ''),
        'image_path': row['image_path'],
        'cancer': int(row['cancer']),
        'classe': cls(int(row['cancer'])),
        **red_hot_metrics(rgb, mask),
    }


def save_panel(df: pd.DataFrame, lut: np.ndarray) -> None:
    panel_dir = OUT_DIR / 'comparacoes_vermelho_quente'
    panel_dir.mkdir(parents=True, exist_ok=True)
    chosen = []
    for c in ['sem_cancer', 'cancer_verdadeiro']:
        chosen.extend(df[df['classe'] == c].sort_values('area_vermelho_dominante', ascending=False).head(MAX_COMPARISON_PER_CLASS).to_dict('records'))
    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(11, max(3, n * 2.2)))
    if n == 1:
        axes = np.array([axes])
    for axrow, item in zip(axes, chosen):
        img = read_rsna_image(Path(item['image_path']), 'mammography')
        gray = normalize(img)
        mask = breast_mask(gray)
        rgb = apply_lut(gray, lut)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        red_dom = (r > 160) & (r > g * 1.15) & (r > b * 1.15) & mask
        overlay = rgb.copy()
        overlay[red_dom] = [255, 0, 0]
        axrow[0].imshow((gray * 255).astype(np.uint8), cmap='gray')
        axrow[0].set_title(f"Original - {item['classe']}")
        axrow[0].axis('off')
        axrow[1].imshow(rgb)
        axrow[1].set_title('AlizaMS RainbowB')
        axrow[1].axis('off')
        axrow[2].imshow(overlay)
        axrow[2].set_title(f"Vermelho {item['area_vermelho_dominante']:.2%}")
        axrow[2].axis('off')
    plt.tight_layout()
    fig.savefig(panel_dir / 'painel_top_vermelho_quente.png', dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    split = pd.read_csv(SPLIT_CSV)
    rows = []
    total = len(split)
    for i, (_, row) in enumerate(split.iterrows(), start=1):
        rows.append(analyze_row(row, lut))
        if i % 50 == 0 or i == total:
            print(f'Processadas {i}/{total} imagens')
    df = pd.DataFrame(rows)
    summary = df.groupby('classe')[[
        'area_vermelho_dominante', 'area_vermelho_forte', 'score_vermelho_medio',
        'score_vermelho_p95', 'canal_r_medio', 'canal_g_medio', 'canal_b_medio'
    ]].mean().reset_index()
    counts = df.assign(vermelho_maior_5pct=df['area_vermelho_dominante'] > 0.05).groupby('classe')['vermelho_maior_5pct'].mean().reset_index()
    summary = summary.merge(counts, on='classe')
    df.to_csv(OUT_DIR / 'vermelho_quente_por_imagem.csv', index=False)
    summary.to_csv(OUT_DIR / 'resumo_vermelho_quente_por_classe.csv', index=False)

    ax = summary.set_index('classe')[['area_vermelho_dominante', 'area_vermelho_forte', 'score_vermelho_medio']].plot(kind='bar', figsize=(9, 5))
    ax.set_title('Vermelho quente com LUT AlizaMS RainbowB')
    ax.set_ylabel('Media por imagem')
    ax.grid(axis='y', alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'barras_vermelho_quente_por_classe.png', dpi=160)
    plt.close()

    save_panel(df, lut)
    (OUT_DIR / 'resumo_vermelho_quente.json').write_text(json.dumps(summary.to_dict('records'), indent=2), encoding='utf-8')
    print('\nResumo vermelho quente:')
    print(summary.to_string(index=False))
    print(f'\nPainel: {OUT_DIR / "comparacoes_vermelho_quente" / "painel_top_vermelho_quente.png"}')
    print(f'Arquivos salvos em: {OUT_DIR}')


if __name__ == '__main__':
    main()
