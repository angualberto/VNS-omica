#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from lidc_sistema_integrado import read_rsna_image

ROOT = Path('/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem')
RUN_DIR = ROOT / 'sistema_integrado_rsna_csv_treino'
SPLIT_CSV = RUN_DIR / 'split_por_series_uid.csv'
ALIZAMS_LUTS = Path('/tmp/AlizaMS/common/luts.h')
OUT_DIR = RUN_DIR / 'alizams_cores_derivadas_gradiente_wavelet_100l'
IMAGE_SIZE = 256
LUT_NAME = 'black_rainbow_lut'
WAVELET = 'db4'
WAVELET_LEVEL = 4
N_LEVELS = 100
TOP_GRADIENT_Q = 0.80
MAX_PANEL_PER_CLASS = 6

COLOR_FAMILIES = {
    'vermelho_rosado': {
        'label': 'Vermelho / Rosado',
        'range_note': 'sangue recente rico em oxigenio',
        'color': '#e53935',
    },
    'roxo_azulado': {
        'label': 'Roxo / Azulado',
        'range_note': 'hemoglobina perdendo oxigenio',
        'color': '#3949ab',
    },
    'esverdeado': {
        'label': 'Esverdeado',
        'range_note': 'biliverdina',
        'color': '#43a047',
    },
    'amarelado_marrom': {
        'label': 'Amarelado / Marrom',
        'range_note': 'bilirrubina/fase final',
        'color': '#f9a825',
    },
}


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


def gradient_maps(gray01: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.GaussianBlur(grad, (3, 3), 0)
    valid = grad[mask]
    if valid.size == 0:
        return mask.astype(np.float32), mask
    grad_norm = grad / (float(valid.max()) + 1e-6)
    threshold = float(np.quantile(valid, TOP_GRADIENT_Q))
    top_mask = mask & (grad >= threshold)
    weights = np.where(mask, 0.10 + grad_norm, 0.0).astype(np.float32)
    return weights, top_mask


def color_family_masks(rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    colorful = mask & (sat >= 0.12) & (val >= 0.08)
    return {
        # vermelho dos dois extremos + rosa/magenta.
        'vermelho_rosado': colorful & ((hue < 18) | (hue >= 300)),
        # azul e roxo juntos.
        'roxo_azulado': colorful & (hue >= 195) & (hue < 300),
        # verde, incluindo verde-amarelado e ciano-esverdeado.
        'esverdeado': colorful & (hue >= 75) & (hue < 195),
        # amarelo, laranja e marrom visual da LUT.
        'amarelado_marrom': colorful & (hue >= 18) & (hue < 75),
    }


def wavelet_profile(weighted_map: np.ndarray) -> dict[str, float]:
    coeffs = pywt.wavedec2(weighted_map.astype(np.float32), wavelet=WAVELET, level=WAVELET_LEVEL, mode='periodization')
    raw: dict[str, float] = {f'LL{WAVELET_LEVEL}': float(np.mean(np.square(coeffs[0])))}
    for level_name, details in zip(range(WAVELET_LEVEL, 0, -1), coeffs[1:]):
        cH, cV, cD = details
        raw[f'L{level_name}_detail'] = float(np.mean(np.square(cH)) + np.mean(np.square(cV)) + np.mean(np.square(cD)))
    total = float(sum(raw.values()) + 1e-12)
    out = {f'wavelet_{k}_prop': float(v / total) for k, v in raw.items()}
    out['wavelet_detail_total_prop'] = float(sum(v for k, v in raw.items() if k != f'LL{WAVELET_LEVEL}') / total)
    out['wavelet_high_L1_prop'] = float(raw.get('L1_detail', 0.0) / total)
    return out


def level_distribution(gray01: np.ndarray, family_mask: np.ndarray, grad_weights: np.ndarray, base_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels = np.floor(gray01 * N_LEVELS).astype(np.int32)
    levels = np.clip(levels, 0, N_LEVELS - 1)
    base_count = np.bincount(levels[base_mask].ravel(), minlength=N_LEVELS).astype(np.float64) + 1e-12
    fam_count = np.bincount(levels[family_mask].ravel(), minlength=N_LEVELS).astype(np.float64)
    area_freq = fam_count / base_count

    base_grad = np.bincount(levels[base_mask].ravel(), weights=grad_weights[base_mask].ravel(), minlength=N_LEVELS).astype(np.float64) + 1e-12
    fam_grad = np.bincount(levels[family_mask].ravel(), weights=grad_weights[family_mask].ravel(), minlength=N_LEVELS).astype(np.float64)
    grad_freq = fam_grad / base_grad
    return area_freq, grad_freq


def cls(cancer: int) -> str:
    return 'cancer_verdadeiro' if int(cancer) else 'sem_cancer'


def analyze_row(row: pd.Series, lut: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    img = read_rsna_image(Path(row['image_path']), 'mammography')
    gray = normalize(img)
    mask = breast_mask(gray)
    rgb = apply_lut(gray, lut)
    grad_weights, top_grad = gradient_maps(gray, mask)
    families = color_family_masks(rgb, mask)

    per_family: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    denom_area = float(mask.sum() + 1e-6)
    denom_grad = float(grad_weights[mask].sum() + 1e-6)
    denom_top = float(top_grad.sum() + 1e-6)

    for family, fam_mask in families.items():
        weighted_map = np.where(fam_mask, grad_weights, 0.0).astype(np.float32)
        area_freq, grad_freq = level_distribution(gray, fam_mask, grad_weights, mask)
        row_base: dict[str, object] = {
            'split': row.get('split', ''),
            'patient_id': row.get('patient_id', ''),
            'image_id': row.get('image_id', ''),
            'image_path': row['image_path'],
            'cancer': int(row['cancer']),
            'classe': cls(int(row['cancer'])),
            'familia_cor': family,
            'familia_cor_label': COLOR_FAMILIES[family]['label'],
            'area_prop': float(fam_mask.sum() / denom_area),
            'grad_prop': float(grad_weights[fam_mask].sum() / denom_grad),
            'top_grad_prop': float((fam_mask & top_grad).sum() / denom_top),
            'grad_medio_na_cor': float(grad_weights[fam_mask].mean()) if fam_mask.any() else 0.0,
            'intensidade_media_na_cor': float(gray[fam_mask].mean()) if fam_mask.any() else 0.0,
            'nivel_100_pico_area': int(np.argmax(area_freq)),
            'nivel_100_pico_grad': int(np.argmax(grad_freq)),
            'freq_pico_area': float(area_freq.max()),
            'freq_pico_grad': float(grad_freq.max()),
        }
        row_base.update(wavelet_profile(weighted_map))
        per_family.append(row_base)

        for level in range(N_LEVELS):
            level_rows.append({
                'split': row.get('split', ''),
                'patient_id': row.get('patient_id', ''),
                'image_id': row.get('image_id', ''),
                'cancer': int(row['cancer']),
                'classe': cls(int(row['cancer'])),
                'familia_cor': family,
                'nivel_100': level,
                'freq_area_no_nivel': float(area_freq[level]),
                'freq_grad_no_nivel': float(grad_freq[level]),
            })
    return per_family, level_rows


def summarize(per_family: pd.DataFrame, level_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = per_family.groupby(['classe', 'familia_cor', 'familia_cor_label'], as_index=False).agg({
        'area_prop': 'mean',
        'grad_prop': 'mean',
        'top_grad_prop': 'mean',
        'grad_medio_na_cor': 'mean',
        'intensidade_media_na_cor': 'mean',
        'nivel_100_pico_area': 'mean',
        'nivel_100_pico_grad': 'mean',
        'wavelet_LL4_prop': 'mean',
        'wavelet_detail_total_prop': 'mean',
        'wavelet_high_L1_prop': 'mean',
    })
    dominant_rows = []
    for classe, g in summary.groupby('classe'):
        dominant_rows.append({
            'classe': classe,
            'familia_area_dominante': g.sort_values('area_prop', ascending=False).iloc[0]['familia_cor'],
            'familia_gradiente_dominante': g.sort_values('grad_prop', ascending=False).iloc[0]['familia_cor'],
            'familia_topgrad_dominante': g.sort_values('top_grad_prop', ascending=False).iloc[0]['familia_cor'],
        })
    dominant = pd.DataFrame(dominant_rows)
    summary = summary.merge(dominant, on='classe', how='left')

    level_summary = level_df.groupby(['classe', 'familia_cor', 'nivel_100'], as_index=False).agg({
        'freq_area_no_nivel': 'mean',
        'freq_grad_no_nivel': 'mean',
    })
    return summary, level_summary


def save_plots(summary: pd.DataFrame, level_summary: pd.DataFrame) -> None:
    colors = [COLOR_FAMILIES[k]['color'] for k in COLOR_FAMILIES]
    for metric, title, filename in [
        ('area_prop', 'Familias de cor por area', 'barras_familias_cor_area.png'),
        ('grad_prop', 'Familias de cor ponderadas por gradiente', 'barras_familias_cor_gradiente.png'),
        ('top_grad_prop', 'Familias de cor no top 20% do gradiente', 'barras_familias_cor_top_gradiente.png'),
        ('wavelet_detail_total_prop', 'Energia wavelet de detalhe por familia de cor', 'barras_wavelet_detalhe_por_familia.png'),
    ]:
        pivot = summary.pivot(index='classe', columns='familia_cor', values=metric).reindex(columns=list(COLOR_FAMILIES))
        ax = pivot.plot(kind='bar', figsize=(11, 5), color=colors)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.25)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=160)
        plt.close()

    for metric, filename in [('freq_area_no_nivel', 'curvas_100_niveis_area.png'), ('freq_grad_no_nivel', 'curvas_100_niveis_gradiente.png')]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, family in zip(axes, COLOR_FAMILIES):
            sub = level_summary[level_summary['familia_cor'] == family]
            for classe, g in sub.groupby('classe'):
                ax.plot(g['nivel_100'], g[metric], label=classe)
            ax.set_title(COLOR_FAMILIES[family]['label'])
            ax.grid(alpha=0.25)
        axes[0].legend()
        fig.supxlabel('Nivel de intensidade 0-99')
        fig.supylabel(metric)
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=160)
        plt.close(fig)


def save_panel(per_family: pd.DataFrame, lut: np.ndarray) -> None:
    panel_dir = OUT_DIR / 'comparacoes_cores_gradiente_wavelet'
    panel_dir.mkdir(parents=True, exist_ok=True)
    selected_images = []
    image_df = per_family.drop_duplicates(['classe', 'patient_id', 'image_id'])
    for classe in ['sem_cancer', 'cancer_verdadeiro']:
        selected_images.extend(image_df[image_df['classe'] == classe].head(MAX_PANEL_PER_CLASS).to_dict('records'))
    if not selected_images:
        return
    n = len(selected_images)
    fig, axes = plt.subplots(n, 4, figsize=(15, max(3, n * 2.25)))
    if n == 1:
        axes = np.array([axes])
    family_colors = {
        'vermelho_rosado': np.array([255, 0, 80], dtype=np.uint8),
        'roxo_azulado': np.array([80, 80, 255], dtype=np.uint8),
        'esverdeado': np.array([0, 255, 0], dtype=np.uint8),
        'amarelado_marrom': np.array([255, 210, 0], dtype=np.uint8),
    }
    for axrow, item in zip(axes, selected_images):
        img = read_rsna_image(Path(item['image_path']), 'mammography')
        gray = normalize(img)
        mask = breast_mask(gray)
        rgb = apply_lut(gray, lut)
        grad_weights, top_grad = gradient_maps(gray, mask)
        families = color_family_masks(rgb, mask)
        heat = np.clip(grad_weights / (grad_weights.max() + 1e-6), 0, 1)
        overlay = rgb.copy()
        for family, fam_mask in families.items():
            overlay[fam_mask & top_grad] = family_colors[family]
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
        axrow[3].set_title('Cores no top gradiente')
        axrow[3].axis('off')
    plt.tight_layout()
    fig.savefig(panel_dir / 'painel_cores_gradiente_wavelet_100l.png', dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    split = pd.read_csv(SPLIT_CSV)
    per_family_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    total = len(split)
    for i, (_, row) in enumerate(split.iterrows(), start=1):
        fam, lev = analyze_row(row, lut)
        per_family_rows.extend(fam)
        level_rows.extend(lev)
        if i % 50 == 0 or i == total:
            print(f'Processadas {i}/{total} imagens')
    per_family = pd.DataFrame(per_family_rows)
    level_df = pd.DataFrame(level_rows)
    summary, level_summary = summarize(per_family, level_df)

    per_family.to_csv(OUT_DIR / 'cores_derivadas_por_imagem_familia.csv', index=False)
    level_df.to_csv(OUT_DIR / 'cores_derivadas_100_niveis_por_imagem.csv', index=False)
    summary.to_csv(OUT_DIR / 'resumo_cores_derivadas_gradiente_wavelet.csv', index=False)
    level_summary.to_csv(OUT_DIR / 'resumo_100_niveis_por_classe.csv', index=False)
    save_plots(summary, level_summary)
    save_panel(per_family, lut)

    payload = {
        'source': 'https://github.com/AlizaMedicalImaging/AlizaMS',
        'lut': 'RainbowB 1536 / black_rainbow_lut',
        'n_levels': N_LEVELS,
        'wavelet': WAVELET,
        'wavelet_level': WAVELET_LEVEL,
        'top_gradient_quantile': TOP_GRADIENT_Q,
        'families': COLOR_FAMILIES,
        'summary': summary.to_dict('records'),
    }
    (OUT_DIR / 'resumo_cores_derivadas_gradiente_wavelet.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('\nResumo por familia de cor:')
    cols = [
        'classe', 'familia_cor', 'area_prop', 'grad_prop', 'top_grad_prop',
        'wavelet_detail_total_prop', 'wavelet_high_L1_prop', 'nivel_100_pico_grad',
        'familia_gradiente_dominante', 'familia_topgrad_dominante',
    ]
    print(summary[cols].to_string(index=False))
    print(f'\nPainel: {OUT_DIR / "comparacoes_cores_gradiente_wavelet" / "painel_cores_gradiente_wavelet_100l.png"}')
    print(f'Arquivos salvos em: {OUT_DIR}')


if __name__ == '__main__':
    main()
