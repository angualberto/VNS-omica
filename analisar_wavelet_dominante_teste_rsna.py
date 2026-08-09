#!/usr/bin/env python3
from __future__ import annotations

import json
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
OUT_DIR = RUN_DIR / 'wavelet_dominante_teste'

WAVELET = 'db4'
LEVEL = 4
IMAGE_SIZE = 256
BANDS = ['LL4', 'LH4', 'HL4', 'HH4', 'LH3', 'HL3', 'HH3', 'LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1']


def normalize_image(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    return (arr - lo) / (hi - lo + 1e-6)


def wavelet_energy_profile(img: np.ndarray) -> tuple[dict[str, float], dict[str, float], str]:
    arr = normalize_image(img)
    coeffs = pywt.wavedec2(arr, wavelet=WAVELET, level=LEVEL, mode='periodization')

    raw = {f'LL{LEVEL}': float(np.mean(np.square(coeffs[0])))}
    # coeffs[1] sao detalhes do nivel mais baixo em frequencia; coeffs[-1] sao os detalhes mais altos.
    for level_name, details in zip(range(LEVEL, 0, -1), coeffs[1:]):
        cH, cV, cD = details
        raw[f'LH{level_name}'] = float(np.mean(np.square(cH)))
        raw[f'HL{level_name}'] = float(np.mean(np.square(cV)))
        raw[f'HH{level_name}'] = float(np.mean(np.square(cD)))

    total = float(sum(raw.values()) + 1e-12)
    props = {name: value / total for name, value in raw.items()}
    predominant = max(props, key=props.get)
    return props, raw, predominant


def analyze_row(row: pd.Series) -> dict[str, object]:
    img = read_rsna_image(Path(row['image_path']), 'mammography')
    props, raw, predominant = wavelet_energy_profile(img)

    low_prop = props['LL4']
    mid_prop = props['LH4'] + props['HL4'] + props['HH4'] + props['LH3'] + props['HL3'] + props['HH3'] + props['LH2'] + props['HL2'] + props['HH2']
    high_prop = props['LH1'] + props['HL1'] + props['HH1']

    result: dict[str, object] = {
        'patient_id': row.get('patient_id', ''),
        'image_id': row.get('image_id', ''),
        'image_path': row['image_path'],
        'cancer': int(row['cancer']),
        'classe': 'cancer_verdadeiro' if int(row['cancer']) == 1 else 'sem_cancer',
        'banda_wavelet_predominante': predominant,
        'prop_low_LL4': float(low_prop),
        'prop_mid_level2': float(mid_prop),
        'prop_high_level1': float(high_prop),
        'high_low_ratio': float(high_prop / (low_prop + 1e-12)),
        'detail_total_prop': float(mid_prop + high_prop),
    }
    for band in BANDS:
        result[f'prop_{band}'] = float(props[band])
        result[f'energy_{band}'] = float(raw[band])
    return result


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cancer_value, group_name in [(0, 'sem_cancer'), (1, 'cancer_verdadeiro')]:
        g = df[df['cancer'] == cancer_value]
        if g.empty:
            continue
        mean_props = {band: float(g[f'prop_{band}'].mean()) for band in BANDS}
        rows.append({
            'classe': group_name,
            'n_imagens': int(len(g)),
            'banda_wavelet_predominante_grupo': max(mean_props, key=mean_props.get),
            'prop_low_LL4_media': float(g['prop_low_LL4'].mean()),
            'prop_mid_level2_media': float(g['prop_mid_level2'].mean()),
            'prop_high_level1_media': float(g['prop_high_level1'].mean()),
            'high_low_ratio_media': float(g['high_low_ratio'].mean()),
            'detail_total_prop_media': float(g['detail_total_prop'].mean()),
            **{f'prop_{band}_media': mean_props[band] for band in BANDS},
        })
    return pd.DataFrame(rows)


def save_plots(per_image: pd.DataFrame, summary: pd.DataFrame) -> None:
    band_cols = [f'prop_{band}' for band in BANDS]
    plot_df = per_image.groupby('classe')[band_cols].mean().T
    plot_df.index = BANDS

    ax = plot_df.plot(kind='bar', figsize=(12, 6), width=0.82)
    ax.set_title('Energia wavelet media por banda - teste RSNA')
    ax.set_xlabel('Banda wavelet')
    ax.set_ylabel('Proporcao media da energia')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(title='Classe')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'barras_energia_wavelet_cancer_vs_sem_cancer.png', dpi=160)
    plt.close()

    labels = ['sem_cancer', 'cancer_verdadeiro']
    data = [per_image.loc[per_image['classe'] == label, 'prop_high_level1'].values for label in labels]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title('Proporcao de alta frequencia wavelet por imagem')
    plt.ylabel('LH1 + HL1 + HH1')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'boxplot_alta_frequencia_wavelet.png', dpi=160)
    plt.close()

    labels = ['sem_cancer', 'cancer_verdadeiro']
    data = [per_image.loc[per_image['classe'] == label, 'high_low_ratio'].values for label in labels]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title('Razao alta/baixa frequencia wavelet por imagem')
    plt.ylabel('alta frequencia / LL4')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'boxplot_razao_alta_baixa_wavelet.png', dpi=160)
    plt.close()

    if len(summary) >= 2:
        stacked = summary.set_index('classe')[['prop_low_LL4_media', 'prop_mid_level2_media', 'prop_high_level1_media']]
        ax = stacked.plot(kind='bar', stacked=True, figsize=(8, 5))
        ax.set_title('Baixa, media e alta frequencia wavelet por classe')
        ax.set_ylabel('Proporcao media')
        ax.set_xlabel('Classe')
        ax.legend(['baixa LL4', 'detalhes niveis 4-2', 'alta nivel 1'])
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'pilha_baixa_media_alta_wavelet.png', dpi=160)
        plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f'Nao encontrei o split de teste: {SPLIT_CSV}')

    split = pd.read_csv(SPLIT_CSV)
    test_df = split[split['split'] == 'test'].copy()
    if test_df.empty:
        raise RuntimeError('O split nao contem imagens de teste.')

    rows = []
    total = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows(), start=1):
        try:
            rows.append(analyze_row(row))
        except Exception as exc:
            rows.append({
                'patient_id': row.get('patient_id', ''),
                'image_id': row.get('image_id', ''),
                'image_path': row.get('image_path', ''),
                'cancer': int(row.get('cancer', -1)),
                'classe': 'erro',
                'erro': str(exc),
            })
        if idx % 25 == 0 or idx == total:
            print(f'Processadas {idx}/{total} imagens')

    per_image = pd.DataFrame(rows)
    ok = per_image[per_image['classe'] != 'erro'].copy()
    if ok.empty:
        raise RuntimeError('Nenhuma imagem foi processada com sucesso.')

    summary = summarize_by_class(ok)
    per_image.to_csv(OUT_DIR / 'wavelet_dominante_por_imagem_teste.csv', index=False)
    summary.to_csv(OUT_DIR / 'resumo_wavelet_dominante_por_classe.csv', index=False)
    save_plots(ok, summary)

    payload = {
        'pywt_version': pywt.__version__,
        'wavelet': WAVELET,
        'level': LEVEL,
        'image_size': IMAGE_SIZE,
        'n_test_images': int(total),
        'n_success': int(len(ok)),
        'n_errors': int((per_image['classe'] == 'erro').sum()),
        'summary': summary.to_dict(orient='records'),
        'outputs': {
            'per_image_csv': str(OUT_DIR / 'wavelet_dominante_por_imagem_teste.csv'),
            'summary_csv': str(OUT_DIR / 'resumo_wavelet_dominante_por_classe.csv'),
        },
    }
    (OUT_DIR / 'resumo_wavelet_dominante.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('\nResumo wavelet dominante por classe:')
    cols = [
        'classe', 'n_imagens', 'banda_wavelet_predominante_grupo',
        'prop_low_LL4_media', 'prop_mid_level2_media', 'prop_high_level1_media',
        'high_low_ratio_media', 'detail_total_prop_media',
    ]
    print(summary[cols].to_string(index=False))
    print(f'\nArquivos salvos em: {OUT_DIR}')


if __name__ == '__main__':
    main()
