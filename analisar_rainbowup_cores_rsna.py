#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
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
OUT_DIR = RUN_DIR / 'rainbowup_cores_todas_imagens'
FILTER_SRC = Path('/tmp/rainbowup/filter.png')
FILTER_LOCAL = OUT_DIR / 'rainbowup_filter.png'
IMAGE_SIZE = 256
ALPHA = 130
SAVE_FILTERED_IMAGES = True
MAX_COMPARISON_PER_CLASS = 6

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


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    arr = (arr - lo) / (hi - lo + 1e-6)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def breast_mask(gray_u8: np.ndarray) -> np.ndarray:
    arr = gray_u8.astype(np.float32) / 255.0
    mask = arr > max(0.03, float(np.quantile(arr, 0.08)))
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return arr > 0.03
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = labels == largest
    if clean.mean() < 0.01:
        return arr > 0.03
    return clean


def apply_rainbowup(gray_u8: np.ndarray, filter_img: Image.Image) -> np.ndarray:
    base = Image.fromarray(np.stack([gray_u8, gray_u8, gray_u8], axis=-1), mode='RGB').convert('RGBA')
    resized_filter = filter_img.resize(base.size)
    return np.asarray(Image.alpha_composite(base, resized_filter).convert('RGB'))


def rgb_to_hsv_metrics(rgb: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], str, str]:
    pixels = rgb[mask]
    if pixels.size == 0:
        pixels = rgb.reshape(-1, 3)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue_deg = hsv[:, 0].astype(np.float32) * 2.0
    sat = hsv[:, 1].astype(np.float32) / 255.0
    val = hsv[:, 2].astype(np.float32) / 255.0

    # Peso por saturacao e brilho para a cor aplicada nao ser dominada por pixels escuros.
    weights = np.clip(sat * val, 1e-4, None)
    total = float(weights.sum() + 1e-12)
    metrics = {
        'hue_medio_graus': float(np.average(hue_deg, weights=weights)),
        'saturacao_media': float(np.average(sat, weights=weights)),
        'brilho_medio': float(np.average(val, weights=weights)),
    }

    # Faixas circulares de matiz. Vermelho pega os dois extremos do circulo.
    ranges = {
        'vermelho': ((hue_deg < 15) | (hue_deg >= 345)),
        'laranja': ((hue_deg >= 15) & (hue_deg < 45)),
        'amarelo': ((hue_deg >= 45) & (hue_deg < 75)),
        'verde': ((hue_deg >= 75) & (hue_deg < 165)),
        'ciano': ((hue_deg >= 165) & (hue_deg < 195)),
        'azul': ((hue_deg >= 195) & (hue_deg < 255)),
        'roxo': ((hue_deg >= 255) & (hue_deg < 300)),
        'rosa': ((hue_deg >= 300) & (hue_deg < 345)),
    }
    for name, sel in ranges.items():
        metrics[f'freq_{name}'] = float(weights[sel].sum() / total)
    dominant = max([name for name, _ in COLOR_BINS], key=lambda name: metrics[f'freq_{name}'])
    second = sorted([name for name, _ in COLOR_BINS], key=lambda name: metrics[f'freq_{name}'], reverse=True)[1]
    return metrics, dominant, second


def class_name(cancer: int) -> str:
    return 'cancer_verdadeiro' if int(cancer) == 1 else 'sem_cancer'


def analyze_row(row: pd.Series, filter_img: Image.Image, filtered_dir: Path) -> dict[str, object]:
    img = read_rsna_image(Path(row['image_path']), 'mammography')
    gray = normalize_to_uint8(img)
    mask = breast_mask(gray)
    rainbow = apply_rainbowup(gray, filter_img)
    metrics, dominant, second = rgb_to_hsv_metrics(rainbow, mask)

    if SAVE_FILTERED_IMAGES:
        cls = class_name(int(row['cancer']))
        out_img = filtered_dir / cls / f"{row['patient_id']}_{row['image_id']}_rainbowup.png"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rainbow).save(out_img)
    else:
        out_img = ''

    return {
        'split': row.get('split', ''),
        'patient_id': row.get('patient_id', ''),
        'image_id': row.get('image_id', ''),
        'image_path': row['image_path'],
        'cancer': int(row['cancer']),
        'classe': class_name(int(row['cancer'])),
        'cor_predominante': dominant,
        'segunda_cor': second,
        'filtered_path': str(out_img),
        **metrics,
    }


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    freq_cols = [f'freq_{name}' for name, _ in COLOR_BINS]
    for cls, g in per_image.groupby('classe'):
        means = {col: float(g[col].mean()) for col in freq_cols}
        rows.append({
            'classe': cls,
            'n_imagens': int(len(g)),
            'cor_predominante_grupo': max(COLOR_BINS, key=lambda item: means[f'freq_{item[0]}'])[0],
            'hue_medio_graus': float(g['hue_medio_graus'].mean()),
            'saturacao_media': float(g['saturacao_media'].mean()),
            'brilho_medio': float(g['brilho_medio'].mean()),
            **{f'{col}_media': means[col] for col in freq_cols},
        })
    return pd.DataFrame(rows)


def save_summary_plots(per_image: pd.DataFrame, summary: pd.DataFrame) -> None:
    freq_cols = [f'freq_{name}' for name, _ in COLOR_BINS]
    plot_df = per_image.groupby('classe')[freq_cols].mean().T
    plot_df.index = [name for name, _ in COLOR_BINS]
    ax = plot_df.plot(kind='bar', figsize=(12, 6), width=0.82, color=['#4c78a8', '#e45756'])
    ax.set_title('Cores predominantes apos filtro rainbowup')
    ax.set_xlabel('Cor')
    ax.set_ylabel('Frequencia media ponderada')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(title='Classe')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'barras_cores_predominantes_por_classe.png', dpi=160)
    plt.close()

    for color_name, color_hex in COLOR_BINS:
        plt.figure(figsize=(8, 5))
        labels = ['sem_cancer', 'cancer_verdadeiro']
        data = [per_image.loc[per_image['classe'] == label, f'freq_{color_name}'].values for label in labels]
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.title(f'Frequencia da cor {color_name} apos rainbowup')
        plt.ylabel('Frequencia ponderada')
        plt.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'boxplot_freq_{color_name}.png', dpi=140)
        plt.close()

    counts = pd.crosstab(per_image['classe'], per_image['cor_predominante'], normalize='index')
    counts = counts.reindex(columns=[name for name, _ in COLOR_BINS], fill_value=0.0)
    ax = counts.plot(kind='bar', stacked=True, figsize=(10, 5), color=[hex_ for _, hex_ in COLOR_BINS])
    ax.set_title('Cor dominante por imagem apos rainbowup')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Proporcao de imagens')
    ax.legend(title='Cor', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pilha_cor_dominante_por_imagem.png', dpi=160)
    plt.close()


def save_comparison_panels(per_image: pd.DataFrame, filter_img: Image.Image) -> None:
    panel_dir = OUT_DIR / 'comparacoes_original_vs_rainbowup'
    panel_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for cls in ['sem_cancer', 'cancer_verdadeiro']:
        g = per_image[per_image['classe'] == cls].head(MAX_COMPARISON_PER_CLASS)
        selected.extend(g.to_dict(orient='records'))

    n = len(selected)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(8, max(3, n * 2.2)))
    if n == 1:
        axes = np.array([axes])
    for axrow, item in zip(axes, selected):
        img = read_rsna_image(Path(item['image_path']), 'mammography')
        gray = normalize_to_uint8(img)
        rainbow = apply_rainbowup(gray, filter_img)
        axrow[0].imshow(gray, cmap='gray')
        axrow[0].set_title(f"Original - {item['classe']}")
        axrow[0].axis('off')
        axrow[1].imshow(rainbow)
        axrow[1].set_title(f"Rainbowup - cor {item['cor_predominante']}")
        axrow[1].axis('off')
    plt.tight_layout()
    fig.savefig(panel_dir / 'painel_comparacao_original_vs_rainbowup.png', dpi=160)
    plt.close(fig)

    for idx, item in enumerate(selected, start=1):
        img = read_rsna_image(Path(item['image_path']), 'mammography')
        gray = normalize_to_uint8(img)
        rainbow = apply_rainbowup(gray, filter_img)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(rainbow)
        axes[1].set_title(f"Rainbowup - {item['cor_predominante']}")
        axes[1].axis('off')
        fig.suptitle(f"{item['classe']} | patient {item['patient_id']} image {item['image_id']}")
        plt.tight_layout()
        fig.savefig(panel_dir / f'comparacao_{idx:02d}_{item["classe"]}.png', dpi=160)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FILTER_SRC.exists():
        raise FileNotFoundError(f'Filtro rainbowup nao encontrado: {FILTER_SRC}. Rode: git clone https://github.com/mtfront/rainbowup.git /tmp/rainbowup')
    shutil.copy2(FILTER_SRC, FILTER_LOCAL)
    filter_img = Image.open(FILTER_LOCAL).convert('RGBA')
    filter_img.putalpha(ALPHA)

    split = pd.read_csv(SPLIT_CSV)
    rows = []
    filtered_dir = OUT_DIR / 'imagens_filtradas_256'
    total = len(split)
    for idx, (_, row) in enumerate(split.iterrows(), start=1):
        try:
            rows.append(analyze_row(row, filter_img, filtered_dir))
        except Exception as exc:
            rows.append({
                'split': row.get('split', ''),
                'patient_id': row.get('patient_id', ''),
                'image_id': row.get('image_id', ''),
                'image_path': row.get('image_path', ''),
                'cancer': int(row.get('cancer', -1)),
                'classe': 'erro',
                'erro': str(exc),
            })
        if idx % 50 == 0 or idx == total:
            print(f'Processadas {idx}/{total} imagens')

    per_image = pd.DataFrame(rows)
    ok = per_image[per_image['classe'] != 'erro'].copy()
    if ok.empty:
        raise RuntimeError('Nenhuma imagem processada com sucesso.')
    summary = summarize(ok)

    per_image.to_csv(OUT_DIR / 'rainbowup_cores_por_imagem.csv', index=False)
    summary.to_csv(OUT_DIR / 'resumo_rainbowup_cores_por_classe.csv', index=False)
    save_summary_plots(ok, summary)
    save_comparison_panels(ok, filter_img)

    payload = {
        'source': 'https://github.com/mtfront/rainbowup',
        'filter': str(FILTER_LOCAL),
        'alpha': ALPHA,
        'image_size': IMAGE_SIZE,
        'n_images': int(total),
        'n_success': int(len(ok)),
        'n_errors': int((per_image['classe'] == 'erro').sum()),
        'summary': summary.to_dict(orient='records'),
    }
    (OUT_DIR / 'resumo_rainbowup_cores.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('\nResumo cores rainbowup por classe:')
    print(summary.to_string(index=False))
    print(f'\nPainel comparativo: {OUT_DIR / "comparacoes_original_vs_rainbowup" / "painel_comparacao_original_vs_rainbowup.png"}')
    print(f'Arquivos salvos em: {OUT_DIR}')


if __name__ == '__main__':
    main()
