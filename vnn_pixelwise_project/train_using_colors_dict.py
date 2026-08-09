#!/usr/bin/env python3
"""Treina um classificador simples usando features de cor já computadas (dicionário CSV).

Uso: python train_using_colors_dict.py --cases-csv <cases.csv> --colors-csv <colors.csv> --out <outdir>
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cases-csv', required=True)
    p.add_argument('--colors-csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()


def extract_image_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = pd.read_csv(args.cases_csv)
    if 'cancer' in cases.columns:
        cases = cases.rename(columns={'cancer': 'label'})
    cases['image_id'] = cases['image_path'].map(extract_image_id).astype(str)

    colors = pd.read_csv(args.colors_csv)
    # ensure keys
    if 'image_id' not in colors.columns:
        if 'image_path' in colors.columns:
            colors['image_id'] = colors['image_path'].map(extract_image_id)
        else:
            raise SystemExit('colors CSV must have image_id or image_path')
    else:
        # if image_id column exists but is empty, try to fill from image_path
        if colors['image_id'].isna().all() and 'image_path' in colors.columns:
            colors['image_id'] = colors['image_path'].map(extract_image_id)
    # coerce to string for reliable merging
    colors['image_id'] = colors['image_id'].astype(str)

    df = cases.merge(colors, on='image_id', how='inner')
    print('Merged rows:', len(df))

    feature_cols = [c for c in df.columns if c.startswith('freq_') or c == 'hue_medio' or c.startswith('freq')]
    if not feature_cols:
        # fallback: use all numeric columns except label
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'label']

    X = df[feature_cols].fillna(0).astype(float).to_numpy()
    y = df['label'].astype(int).to_numpy()

    Xtr, Xte, ytr, yte, df_tr, df_te = train_test_split(X, y, df, test_size=args.test_size, random_state=args.random_state, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr_s, ytr)

    p_tr = model.predict_proba(Xtr_s)[:,1]
    p_te = model.predict_proba(Xte_s)[:,1]

    metrics = {
        'train_roc_auc': roc_auc_score(ytr, p_tr) if len(np.unique(ytr))>1 else float('nan'),
        'train_pr_auc': average_precision_score(ytr, p_tr) if len(np.unique(ytr))>1 else float('nan'),
        'train_acc': accuracy_score(ytr, (p_tr>0.5).astype(int)),
        'test_roc_auc': roc_auc_score(yte, p_te) if len(np.unique(yte))>1 else float('nan'),
        'test_pr_auc': average_precision_score(yte, p_te) if len(np.unique(yte))>1 else float('nan'),
        'test_acc': accuracy_score(yte, (p_te>0.5).astype(int)),
    }

    pd.DataFrame([metrics]).to_csv(os.path.join(args.out, 'colors_model_metrics.csv'), index=False)

    out_pred = df_te.copy()
    out_pred['score_colors_model'] = p_te
    cols_want = ['patient_id','image_id','image_path','label','score_colors_model']
    cols = [c for c in cols_want if c in out_pred.columns]
    out_pred[cols].to_csv(os.path.join(args.out, 'colors_model_predictions.csv'), index=False)

    # save scaler and model via numpy
    np.save(os.path.join(args.out, 'colors_model_coef.npy'), model.coef_)
    np.save(os.path.join(args.out, 'colors_model_intercept.npy'), model.intercept_)
    print('Saved metrics and predictions to', args.out)


if __name__ == '__main__':
    main()
