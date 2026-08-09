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
OUT_DIR = RUN_DIR / 'alizams_rainbowb_gradiente_cores'
IMAGE_SIZE = 256
LUT_NAME = 'black_rainbow_lut'
LUT_LABEL = 'AlizaMS RainbowB 1536'
TOP_GRADIENT_Q = 0.80
MAX_PANEL_PER_CLASS = 6

COLOR_BINS = [
    ('vermelho', '#e53935'),
    ('laranja', '#fb8c00'),
    ('amarelo', '#fdd835'),
    ('verde', '#43a047'),
    ('ciano', '#00acc1'),
    ('azul', '#1e88e5'),
    ('roxo', '#8e24aa'),
    ('rosa', '#d81b60'),
]


def load_lut() -> np.ndarray:
    text = ALIZAMS_LUTS.read_text(encoding='utf-8', errors='ignore')
    m = re.search(rf'constexpr unsigned char {LUT_NAME}\[[^\]]+\]\s*=\s*\{{(.*?)\}};', text, re.S)
    if not m:
        raise RuntimeError(f'LUT {LUT_NAME} nao encontrada em {ALIZAMS_LUTS}')
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


def gradient_weight(gray01: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.GaussianBlur(grad, (3, 3), 0)
    valid = grad[mask]
    if valid.size == 0:
        weights = mask.astype(np.float32)
        return weights, mask
    grad_norm = grad / (float(valid.max()) + 1e-6)
    threshold = float(np.quantile(valid, TOP_GRADIENT_Q))
    top_mask = mask & (grad >= threshold)
    weights = np.where(mask, 0.15 + grad_norm, 0.0).astype(np.float32)
    return weights, top_mask


def color_metrics(rgb: np.ndarray, weights: np.ndarray, top_mask: np.ndarray) -> tuple[dict[str, float], str, str]:
    flat_rgb = rgb.reshape(-1, 3).astype(np.uint8)
    flat_w = weights.reshape(-1).astype(np.float32)
    keep = flat_w > 0
    if not np.any(keep):
        keep = np.ones_like(flat_w, dtype=bool)
        flat_w = np.ones_like(flat_w, dtype=np.float32)
    hsv = cv2.cvtColor(flat_rgb[keep].reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0].astype(np.float32) * 2.0
    sat = hsv[:, 1].astype(np.float32) / 255.0
    val = hsv[:, 2].astype(np.float32) / 255.0
    w = flat_w[keep] * np.clip(sat, 0.05, None) * np.clip(val, 0.05, None)
    total = float(w.sum() + 1e-12)

    ranges = {
        'vermelho': ((hue < 15) | (hue >= 345)),
        'laranja': ((hue >= 15) & (hue < 45)),
        'amarelo': ((hue >= 45) & (hue < 75)),
        'verde': ((hue >= 75) & (hue < 165)),
        'ciano': ((hue >= 165) & (hue < 195)),
        'azul': ((hue >= 195) & (hue < 255)),
        'roxo': ((hue >= 255) & (hue < 300)),
        'rosa': ((hue >= 300) & (hue < 345)),
    }
    metrics: dict[str, float] = {
        'hue_gradiente_medio': float(np.average(hue, weights=w)),
        'saturacao_gradiente_media': float(np.average(sat, weights=w)),
        'brilho_gradiente_medio': float(np.average(val, weights=w)),
        'area_top_gradiente': float(top_mask.mean()),
    }
    for name, sel in ranges.items():
        metrics[f'freq_grad_{name}'] = float(w[sel].sum() / total)

    # Tambem mede so o top 20% de gradiente, sem suavizar pelo restante.
    top_w = top_mask.reshape(-1).astype(np.float32)
    top_keep = top_w > 0
    if np.any(top_keep):
        top_hsv = cv2.cvtColor(flat_rgb[top_keep].reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        top_hue = top_hsv[:, 0].astype(np.float32) * 2.0
        top_sat = top_hsv[:, 1].astype(np.float32) / 255.0
        top_val = top_hsv[:, 2].astype(np.float32) / 255.0
        top_weight = np.clip(top_sat * top_val, 1e-4, None)
        top_total = float(top_weight.sum() + 1e-12)
        for name, sel_range in ranges.items():
            if name == 'vermelho':
                sel = (top_hue < 15) | (top_hue >= 345)
            elif name == 'laranja':
                sel = (top_hue >= 15) & (top_hue < 45)
            elif name == 'amarelo':
                sel = (top_hue >= 45) & (top_hue < 75)
            elif name == 'verde':
                sel = (top_hue >= 75) & (top_hue < 165)
            elif name == 'ciano':
                sel = (top_hue >= 165) & (top_hue < 195)
            elif name == 'azul':
                sel = (top_hue >= 195) & (top_hue < 255)
            elif name == 'roxo':
                sel = (top_hue >= 255) & (top_hue < 300)
            else:
                sel = (top_hue >= 300) & (top_hue < 345)
            metrics[f'freq_topgrad_{name}'] = float(top_weight[sel].sum() / top_total)
    else:
        for name, _ in COLOR_BINS:
            metrics[f'freq_topgrad_{name}'] = 0.0

    ordered = sorted([name for name, _ in COLOR_BINS], key=lambda n: metrics[f'freq_grad_{n}'], reverse=True)
    return metrics, ordered[0], ordered[1]


def cls(cancer: int) -> str:
    return 'cancer_verdadeiro' if int(cancer) else 'sem_cancer'


def analyze_row(row: pd.Series, lut: np.ndarray) -> dict[str, object]:
    img = read_rsna_image(Path(row['image_path']), 'mammography')
    gray = normalize(img)
    mask = breast_mask(gray)
    rgb = apply_lut(gray, lut)
    weights, top_mask = gradient_weight(gray, mask)
    metrics, dominant, second = color_metrics(rgb, weights, top_mask)
    return {
        'split': row.get('split', ''),
        'patient_id': row.get('patient_id', ''),
        'image_id': row.get('image_id', ''),
        'image_path': row['image_path'],
        'cancer': int(row['cancer']),
        'classe': cls(int(row['cancer'])),
        'cor_predominante_gradiente': dominant,
        'segunda_cor_gradiente': second,
        **metrics,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    freq_cols = [f'freq_grad_{name}' for name, _ in COLOR_BINS]
    top_cols = [f'freq_topgrad_{name}' for name, _ in COLOR_BINS]
    for classe, g in df.groupby('classe'):
        means = {col: float(g[col].mean()) for col in freq_cols + top_cols}
        rows.append({
            'classe': classe,
            'n_imagens': int(len(g)),
            'cor_predominante_gradiente_grupo': max(COLOR_BINS, key=lambda item: means[f'freq_grad_{item[0]}'])[0],
            'cor_predominante_topgrad_grupo': max(COLOR_BINS, key=lambda item: means[f'freq_topgrad_{item[0]}'])[0],
            'hue_gradiente_medio': float(g['hue_gradiente_medio'].mean()),
            'area_top_gradiente_media': float(g['area_top_gradiente'].mean()),
            **{f'{col}_media': means[col] for col in freq_cols + top_cols},
        })
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    freq_cols = [f'freq_grad_{name}' for name, _ in COLOR_BINS]
    plot_df = df.groupby('classe')[freq_cols].mean().T
    plot_df.index = [name for name, _ in COLOR_BINS]
    ax = plot_df.plot(kind='bar', figsize=(12, 6), width=0.82, color=['#4c78a8', '#e45756'])
    ax.set_title('Cor predominante ponderada por gradiente - AlizaMS RainbowB')
    ax.set_xlabel('Cor')
    ax.set_ylabel('Frequencia media ponderada por gradiente')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(title='Classe')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'barras_cores_por_gradiente.png', dpi=160)
    plt.close()

    top_cols = [f'freq_topgrad_{name}' for name, _ in COLOR_BINS]
    top_df = df.groupby('classe')[top_cols].mean().T
    top_df.index = [name for name, _ in COLOR_BINS]
    ax = top_df.plot(kind='bar', figsize=(12, 6), width=0.82, color=['#4c78a8', '#e45756'])
    ax.set_title('Cor no top 20% de gradiente - AlizaMS RainbowB')
    ax.set_xlabel('Cor')
    ax.set_ylabel('Frequencia media')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(title='Classe')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'barras_cores_top_gradiente.png', dpi=160)
    plt.close()

    counts = pd.crosstab(df['classe'], df['cor_predominante_gradiente'], normalize='index')
    counts = counts.reindex(columns=[name for name, _ in COLOR_BINS], fill_value=0.0)
    ax = counts.plot(kind='bar', stacked=True, figsize=(10, 5), color=[hex_ for _, hex_ in COLOR_BINS])
    ax.set_title('Cor dominante por imagem ponderada por gradiente')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Proporcao de imagens')
    ax.legend(title='Cor', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pilha_cor_dominante_gradiente.png', dpi=160)
    plt.close()


def save_panel(df: pd.DataFrame, lut: np.ndarray) -> None:
    panel_dir = OUT_DIR / 'comparacoes_gradiente'
    panel_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for classe in ['sem_cancer', 'cancer_verdadeiro']:
        selected.extend(df[df['classe'] == classe].head(MAX_PANEL_PER_CLASS).to_dict('records'))
    n = len(selected)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(14, max(3, n * 2.2)))
    if n == 1:
        axes = np.array([axes])
    for axrow, item in zip(axes, selected):
        img = read_rsna_image(Path(item['image_path']), 'mammography')
        gray = normalize(img)
        mask = breast_mask(gray)
        rgb = apply_lut(gray, lut)
        weights, top_mask = gradient_weight(gray, mask)
        heat = np.clip(weights / (weights.max() + 1e-6), 0, 1)
        overlay = rgb.copy()
        overlay[top_mask] = np.array([255, 255, 255], dtype=np.uint8)
        axrow[0].imshow((gray * 255).astype(np.uint8), cmap='gray')
        axrow[0].set_title(f"Original - {item['classe']}")
        axrow[0].axis('off')
        axrow[1].imshow(rgb)
        axrow[1].set_title('AlizaMS RainbowB')
        axrow[1].axis('off')
        axrow[2].imshow(heat, cmap='inferno')
        axrow[2].set_title('Gradiente')
        axrow[2].axis('off')
        axrow[3].imshow(overlay)
        axrow[3].set_title(f"Cor: {item['cor_predominante_gradiente']}")
        axrow[3].axis('off')
    plt.tight_layout()
    fig.savefig(panel_dir / 'painel_gradiente_cores_alizams.png', dpi=160)
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
    summary = summarize(df)
    df.to_csv(OUT_DIR / 'cores_gradiente_por_imagem.csv', index=False)
    summary.to_csv(OUT_DIR / 'resumo_cores_gradiente_por_classe.csv', index=False)
    save_plots(df, summary)
    save_panel(df, lut)
    (OUT_DIR / 'resumo_cores_gradiente.json').write_text(json.dumps({
        'source': 'https://github.com/AlizaMedicalImaging/AlizaMS',
        'lut': LUT_LABEL,
        'top_gradient_quantile': TOP_GRADIENT_Q,
        'summary': summary.to_dict('records'),
    }, indent=2), encoding='utf-8')
    print('\nResumo cores ponderadas por gradiente:')
    cols = ['classe', 'n_imagens', 'cor_predominante_gradiente_grupo', 'cor_predominante_topgrad_grupo',
            'freq_grad_vermelho_media', 'freq_grad_verde_media', 'freq_grad_azul_media', 'freq_grad_roxo_media',
            'freq_topgrad_vermelho_media', 'freq_topgrad_verde_media', 'freq_topgrad_azul_media']
    print(summary[cols].to_string(index=False))
    print(f'\nPainel: {OUT_DIR / "comparacoes_gradiente" / "painel_gradiente_cores_alizams.png"}')
    print(f'Arquivos salvos em: {OUT_DIR}')


if __name__ == '__main__':
    main()
