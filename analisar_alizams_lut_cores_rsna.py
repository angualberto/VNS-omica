#!/usr/bin/env python3
from __future__ import annotations

import json
import re
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
ALIZAMS_ROOT = Path('/tmp/AlizaMS')
ALIZAMS_LUTS = ALIZAMS_ROOT / 'common' / 'luts.h'
OUT_DIR = RUN_DIR / 'alizams_rainbowb_cores_todas_imagens'
IMAGE_SIZE = 256
LUT_NAME = 'black_rainbow_lut'
LUT_LABEL = 'AlizaMS RainbowB 1536'
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


def load_alizams_lut(name: str) -> np.ndarray:
    text = ALIZAMS_LUTS.read_text(encoding='utf-8', errors='ignore')
    m = re.search(rf'constexpr unsigned char {re.escape(name)}\[[^\]]+\]\s*=\s*\{{(.*?)\}};', text, re.S)
    if not m:
        raise RuntimeError(f'LUT {name} nao encontrada em {ALIZAMS_LUTS}')
    nums = [int(x) for x in re.findall(r'\d+', m.group(1))]
    arr = np.asarray(nums, dtype=np.uint8)
    if arr.size % 3 != 0:
        raise RuntimeError(f'LUT {name} tem tamanho invalido: {arr.size}')
    return arr.reshape(-1, 3)


def normalize_to_float(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def breast_mask(gray01: np.ndarray) -> np.ndarray:
    mask = gray01 > max(0.03, float(np.quantile(gray01, 0.08)))
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = labels == largest
    if clean.mean() < 0.01:
        return mask
    return clean


def apply_alizams_lut(gray01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    # Equivalente ao ProcessImageThreadLUT do AlizaMS em LINEAR_EXACT:
    # r = intensidade normalizada, z = int(r * tamanho_lut), limitado aos extremos.
    idx = np.floor(gray01 * len(lut)).astype(np.int32)
    idx = np.clip(idx, 0, len(lut) - 1)
    return lut[idx]


def rgb_to_hsv_metrics(rgb: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], str, str]:
    pixels = rgb[mask]
    if pixels.size == 0:
        pixels = rgb.reshape(-1, 3)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue_deg = hsv[:, 0].astype(np.float32) * 2.0
    sat = hsv[:, 1].astype(np.float32) / 255.0
    val = hsv[:, 2].astype(np.float32) / 255.0
    weights = np.clip(sat * val, 1e-4, None)
    total = float(weights.sum() + 1e-12)
    metrics = {
        'hue_medio_graus': float(np.average(hue_deg, weights=weights)),
        'saturacao_media': float(np.average(sat, weights=weights)),
        'brilho_medio': float(np.average(val, weights=weights)),
    }
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
    ordered = sorted([name for name, _ in COLOR_BINS], key=lambda name: metrics[f'freq_{name}'], reverse=True)
    return metrics, ordered[0], ordered[1]


def class_name(cancer: int) -> str:
    return 'cancer_verdadeiro' if int(cancer) == 1 else 'sem_cancer'


def analyze_row(row: pd.Series, lut: np.ndarray, filtered_dir: Path) -> dict[str, object]:
    img = read_rsna_image(Path(row['image_path']), 'mammography')
    gray01 = normalize_to_float(img)
    mask = breast_mask(gray01)
    colored = apply_alizams_lut(gray01, lut)
    metrics, dominant, second = rgb_to_hsv_metrics(colored, mask)

    if SAVE_FILTERED_IMAGES:
        cls = class_name(int(row['cancer']))
        out_img = filtered_dir / cls / f"{row['patient_id']}_{row['image_id']}_alizams_rainbowb.png"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colored).save(out_img)
    else:
        out_img = ''

    return {
        'split': row.get('split', ''),
        'patient_id': row.get('patient_id', ''),
        'image_id': row.get('image_id', ''),
        'image_path': row['image_path'],
        'cancer': int(row['cancer']),
        'classe': class_name(int(row['cancer'])),
        'lut': LUT_LABEL,
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


def save_summary_plots(per_image: pd.DataFrame) -> None:
    freq_cols = [f'freq_{name}' for name, _ in COLOR_BINS]
    plot_df = per_image.groupby('classe')[freq_cols].mean().T
    plot_df.index = [name for name, _ in COLOR_BINS]
    ax = plot_df.plot(kind='bar', figsize=(12, 6), width=0.82, color=['#4c78a8', '#e45756'])
    ax.set_title('Cores predominantes com LUT AlizaMS RainbowB')
    ax.set_xlabel('Cor')
    ax.set_ylabel('Frequencia media ponderada')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(title='Classe')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'barras_cores_predominantes_alizams_rainbowb.png', dpi=160)
    plt.close()

    counts = pd.crosstab(per_image['classe'], per_image['cor_predominante'], normalize='index')
    counts = counts.reindex(columns=[name for name, _ in COLOR_BINS], fill_value=0.0)
    ax = counts.plot(kind='bar', stacked=True, figsize=(10, 5), color=[hex_ for _, hex_ in COLOR_BINS])
    ax.set_title('Cor dominante por imagem com LUT AlizaMS RainbowB')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Proporcao de imagens')
    ax.legend(title='Cor', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pilha_cor_dominante_alizams_rainbowb.png', dpi=160)
    plt.close()


def save_comparison_panels(per_image: pd.DataFrame, lut: np.ndarray) -> None:
    panel_dir = OUT_DIR / 'comparacoes_original_vs_alizams_rainbowb'
    panel_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for cls in ['sem_cancer', 'cancer_verdadeiro']:
        selected.extend(per_image[per_image['classe'] == cls].head(MAX_COMPARISON_PER_CLASS).to_dict(orient='records'))
    if not selected:
        return
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(8, max(3, n * 2.2)))
    if n == 1:
        axes = np.array([axes])
    for axrow, item in zip(axes, selected):
        img = read_rsna_image(Path(item['image_path']), 'mammography')
        gray01 = normalize_to_float(img)
        gray_u8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
        colored = apply_alizams_lut(gray01, lut)
        axrow[0].imshow(gray_u8, cmap='gray')
        axrow[0].set_title(f"Original - {item['classe']}")
        axrow[0].axis('off')
        axrow[1].imshow(colored)
        axrow[1].set_title(f"AlizaMS RainbowB - {item['cor_predominante']}")
        axrow[1].axis('off')
    plt.tight_layout()
    fig.savefig(panel_dir / 'painel_comparacao_original_vs_alizams_rainbowb.png', dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ALIZAMS_LUTS.exists():
        raise FileNotFoundError(f'AlizaMS nao encontrado em {ALIZAMS_ROOT}. Rode: git clone https://github.com/AlizaMedicalImaging/AlizaMS.git /tmp/AlizaMS')
    shutil.copy2(ALIZAMS_LUTS, OUT_DIR / 'alizams_luts.h')
    lut = load_alizams_lut(LUT_NAME)
    split = pd.read_csv(SPLIT_CSV)
    rows = []
    filtered_dir = OUT_DIR / 'imagens_filtradas_256'
    total = len(split)
    for idx, (_, row) in enumerate(split.iterrows(), start=1):
        try:
            rows.append(analyze_row(row, lut, filtered_dir))
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
    per_image.to_csv(OUT_DIR / 'alizams_rainbowb_cores_por_imagem.csv', index=False)
    summary.to_csv(OUT_DIR / 'resumo_alizams_rainbowb_cores_por_classe.csv', index=False)
    save_summary_plots(ok)
    save_comparison_panels(ok, lut)
    (OUT_DIR / 'resumo_alizams_rainbowb_cores.json').write_text(json.dumps({
        'source': 'https://github.com/AlizaMedicalImaging/AlizaMS',
        'lut_label': LUT_LABEL,
        'lut_name': LUT_NAME,
        'lut_size': int(len(lut)),
        'image_size': IMAGE_SIZE,
        'n_images': int(total),
        'n_success': int(len(ok)),
        'n_errors': int((per_image['classe'] == 'erro').sum()),
        'summary': summary.to_dict(orient='records'),
    }, indent=2), encoding='utf-8')

    print('\nResumo cores AlizaMS RainbowB por classe:')
    print(summary.to_string(index=False))
    print(f'\nPainel comparativo: {OUT_DIR / "comparacoes_original_vs_alizams_rainbowb" / "painel_comparacao_original_vs_alizams_rainbowb.png"}')
    print(f'Arquivos salvos em: {OUT_DIR}')


if __name__ == '__main__':
    main()
