#!/usr/bin/env python3
"""Compara frequencias dominantes no conjunto de teste RSNA.

Usa o split gerado pelo treinamento em sistema_integrado_rsna_csv_treino.
Para cada mamografia do teste, calcula a FFT local em janelas e mede qual
banda radial domina. Depois compara cancer=1 contra cancer=0.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lidc_sistema_integrado import read_rsna_image


ROOT = Path('/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem')
RUN_DIR = ROOT / 'sistema_integrado_rsna_csv_treino'
SPLIT_CSV = RUN_DIR / 'split_por_series_uid.csv'
OUT_DIR = RUN_DIR / 'frequencia_dominante_teste'

# 15 bandas uteis: 1 perto de baixa frequencia, 15 perto de alta frequencia.
BAND_LABELS = [f'banda_{i:02d}' for i in range(1, 16)]


def dominant_frequency_grid(img: np.ndarray, size: int = 256, window: int = 32, stride: int = 8):
    arr = cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    arr = (arr - float(arr.min())) / (float(arr.max() - arr.min()) + 1e-6)
    h, w = arr.shape

    yy, xx = np.indices((window, window))
    cy = cx = window // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (rr.max() + 1e-6)
    edges = np.linspace(0.0, 1.0, 17)

    dominant_bins = []
    grid_rows = []
    for y0 in range(0, h - window + 1, stride):
        row_bins = []
        for x0 in range(0, w - window + 1, stride):
            patch = arr[y0:y0 + window, x0:x0 + window]
            patch = patch - patch.mean()
            mag = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
            energy = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                band = mag[(rr >= lo) & (rr < hi)]
                energy.append(float(band.mean()) if band.size else 0.0)
            # Ignora a banda 0/DC. Retorna bandas 1..15.
            dominant = int(np.argmax(energy[1:]) + 1)
            dominant_bins.append(dominant)
            row_bins.append(dominant)
        grid_rows.append(row_bins)

    bins = np.asarray(dominant_bins, dtype=np.int16)
    grid = np.asarray(grid_rows, dtype=np.float32)
    hist = np.bincount(bins, minlength=16)[1:16].astype(np.float64)
    hist_prop = hist / max(float(hist.sum()), 1.0)
    return grid, bins, hist_prop


def summarize_image(row) -> tuple[dict, np.ndarray]:
    img = read_rsna_image(Path(row.image_path), 'mammography')
    grid, bins, hist_prop = dominant_frequency_grid(img)
    mode_band = int(np.argmax(hist_prop) + 1)
    out = {
        'patient_id': str(row.patient_id),
        'image_id': str(row.image_id),
        'laterality': getattr(row, 'laterality', ''),
        'view': getattr(row, 'view', ''),
        'cancer': int(row.cancer),
        'image_path': str(row.image_path),
        'n_janelas': int(len(bins)),
        'banda_dominante_moda': mode_band,
        'freq_dominante_media': float(bins.mean()),
        'freq_dominante_mediana': float(np.median(bins)),
        'freq_dominante_p85': float(np.percentile(bins, 85)),
        'freq_dominante_p95': float(np.percentile(bins, 95)),
        'prop_baixa_b1_b5': float(hist_prop[:5].sum()),
        'prop_media_b6_b10': float(hist_prop[5:10].sum()),
        'prop_alta_b11_b15': float(hist_prop[10:15].sum()),
    }
    for i, value in enumerate(hist_prop, start=1):
        out[f'prop_banda_{i:02d}'] = float(value)
    return out, grid


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cancer, g in df.groupby('cancer'):
        mean_props = np.array([g[f'prop_banda_{i:02d}'].mean() for i in range(1, 16)])
        predominant = int(np.argmax(mean_props) + 1)
        rows.append({
            'grupo': 'cancer_verdadeiro' if int(cancer) == 1 else 'sem_cancer',
            'cancer': int(cancer),
            'n_imagens': int(len(g)),
            'banda_predominante_grupo': predominant,
            'freq_media_media': float(g['freq_dominante_media'].mean()),
            'freq_mediana_media': float(g['freq_dominante_mediana'].mean()),
            'p85_medio': float(g['freq_dominante_p85'].mean()),
            'prop_baixa_b1_b5_media': float(g['prop_baixa_b1_b5'].mean()),
            'prop_media_b6_b10_media': float(g['prop_media_b6_b10'].mean()),
            'prop_alta_b11_b15_media': float(g['prop_alta_b11_b15'].mean()),
            **{f'prop_banda_{i:02d}_media': float(mean_props[i - 1]) for i in range(1, 16)},
        })
    return pd.DataFrame(rows).sort_values('cancer')


def save_plots(per_image: pd.DataFrame, summary: pd.DataFrame, mean_maps: dict[int, np.ndarray]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 16)
    width = 0.38
    for offset, cancer, label, color in [(-width / 2, 0, 'sem cancer', '#2563eb'), (width / 2, 1, 'cancer verdadeiro', '#dc2626')]:
        row = summary[summary['cancer'] == cancer]
        if row.empty:
            continue
        vals = [float(row.iloc[0][f'prop_banda_{i:02d}_media']) for i in range(1, 16)]
        ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.82)
    ax.set_xlabel('Banda dominante local (1=mais baixa, 15=mais alta)')
    ax.set_ylabel('Proporcao media de janelas')
    ax.set_title('Distribuicao de frequencia dominante no teste RSNA')
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'histograma_bandas_cancer_vs_sem_cancer.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        per_image.loc[per_image['cancer'] == 0, 'freq_dominante_media'].to_numpy(),
        per_image.loc[per_image['cancer'] == 1, 'freq_dominante_media'].to_numpy(),
    ]
    ax.boxplot(data, labels=['sem cancer', 'cancer verdadeiro'], showmeans=True)
    ax.set_ylabel('Banda dominante media por imagem')
    ax.set_title('Comparacao da frequencia dominante media')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'boxplot_frequencia_media_cancer_vs_sem_cancer.png', dpi=160)
    plt.close(fig)

    available = [c for c in [0, 1] if c in mean_maps]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        for ax, cancer in zip(axes, available):
            im = ax.imshow(mean_maps[cancer], cmap='turbo', vmin=1, vmax=15)
            ax.set_title('cancer verdadeiro' if cancer == 1 else 'sem cancer')
            ax.axis('off')
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label='Banda dominante media')
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'mapa_medio_frequencia_dominante.png', dpi=160)
        plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f'Nao encontrei o split de teste: {SPLIT_CSV}')

    split = pd.read_csv(SPLIT_CSV)
    test = split[split['split'] == 'test'].copy().reset_index(drop=True)
    if test.empty:
        raise RuntimeError('Nenhuma imagem de teste encontrada no split_por_series_uid.csv')

    rows = []
    grids_by_class: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for i, row in enumerate(test.itertuples(index=False), start=1):
        metrics, grid = summarize_image(row)
        rows.append(metrics)
        grids_by_class[int(row.cancer)].append(grid)
        if i % 25 == 0 or i == len(test):
            print(f'Frequencia dominante teste: {i}/{len(test)}', flush=True)

    per_image = pd.DataFrame(rows)
    summary = group_summary(per_image)
    mean_maps = {
        cancer: np.mean(np.stack(grids), axis=0)
        for cancer, grids in grids_by_class.items()
        if grids
    }

    per_image.to_csv(OUT_DIR / 'frequencia_dominante_por_imagem_teste.csv', index=False)
    summary.to_csv(OUT_DIR / 'resumo_frequencia_dominante_por_classe.csv', index=False)
    save_plots(per_image, summary, mean_maps)

    result = {
        'entrada_split': str(SPLIT_CSV),
        'n_teste': int(len(test)),
        'n_cancer_verdadeiro': int((test['cancer'] == 1).sum()),
        'n_sem_cancer': int((test['cancer'] == 0).sum()),
        'saida': str(OUT_DIR),
        'resumo': summary.to_dict(orient='records'),
    }
    (OUT_DIR / 'resumo_frequencia_dominante.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\nResumo por classe:')
    print(summary[[
        'grupo', 'n_imagens', 'banda_predominante_grupo', 'freq_media_media',
        'prop_baixa_b1_b5_media', 'prop_media_b6_b10_media', 'prop_alta_b11_b15_media'
    ]].to_string(index=False))
    print(f'\nArquivos salvos em: {OUT_DIR}')


if __name__ == '__main__':
    main()
