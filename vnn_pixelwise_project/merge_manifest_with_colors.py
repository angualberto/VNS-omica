#!/usr/bin/env python3
"""Mescla um manifest (shards) com o CSV de cores gerado a partir do dicionário.

Gera um arquivo CSV com as colunas do manifest mais as colunas de cor (freq_*, hue_medio, lut).
Uso: python merge_manifest_with_colors.py --manifest <manifest.csv> --colors-csv <colors.csv> --out <out.csv>
"""
import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--colors-csv', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args()


def extract_image_id(path: str) -> str:
    if pd.isna(path):
        return ''
    return os.path.splitext(os.path.basename(path))[0]


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={'patient_id': str, 'image_id': str})
    colors = pd.read_csv(args.colors_csv)
    if 'image_id' not in colors.columns or colors['image_id'].isna().all():
        if 'image_path' in colors.columns:
            colors['image_id'] = colors['image_path'].map(extract_image_id)
        else:
            raise SystemExit('colors CSV must have image_id or image_path')
    colors['image_id'] = colors['image_id'].astype(str)
    merged = manifest.merge(colors.drop_duplicates(['patient_id', 'image_id']) if 'patient_id' in colors.columns else colors.drop_duplicates(['image_id']), on=['patient_id', 'image_id'] if 'patient_id' in colors.columns else ['image_id'], how='left')
    merged.to_csv(args.out, index=False)
    print('Wrote', args.out, 'rows=', len(merged))


if __name__ == '__main__':
    main()
