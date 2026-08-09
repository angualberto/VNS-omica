#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from lidc_sistema_integrado import read_rsna_image

ROOT = Path('/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem')
RUN_DIR = ROOT / 'sistema_integrado_rsna_csv_treino'
SPLIT_CSV = RUN_DIR / 'split_por_series_uid.csv'
ALIZAMS_LUTS = Path('/tmp/AlizaMS/common/luts.h')
OUT_DIR = RUN_DIR / 'classificador_patches_xgboost_wavelet_gradiente_cor_fft'

IMAGE_SIZE = 256
PATCH_SIZE = 96
PATCH_STRIDE = 48
TOP_PATCHES = 12
LUT_NAME = 'black_rainbow_lut'
WAVELET = 'db4'
WAVELET_LEVEL = 2
RANDOM_STATE = 42

COLOR_FAMILIES = ['vermelho_rosado', 'roxo_azulado', 'esverdeado', 'amarelado_marrom']


def load_lut() -> np.ndarray:
    text = ALIZAMS_LUTS.read_text(encoding='utf-8', errors='ignore')
    m = re.search(rf'constexpr unsigned char {LUT_NAME}\[[^\]]+\]\s*=\s*\{{(.*?)\}};', text, re.S)
    if not m:
        raise RuntimeError(f'LUT {LUT_NAME} nao encontrada em {ALIZAMS_LUTS}')
    nums = [int(x) for x in re.findall(r'\d+', m.group(1))]
    return np.asarray(nums, dtype=np.uint8).reshape(-1, 3)


def normalize_image(img: np.ndarray) -> np.ndarray:
    arr = cv2.resize(img.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    lo, hi = float(arr.min()), float(arr.max())
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)


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


def safe_entropy(values: np.ndarray, bins: int = 32) -> float:
    vals = np.asarray(values, dtype=np.float32).ravel()
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gradient_maps(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def gradient_features(patch: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gx, gy, mag = gradient_maps(patch)
    vals = mag[mask]
    if vals.size == 0:
        vals = mag.ravel()
    top_thr = float(np.quantile(vals, 0.80)) if vals.size else 0.0
    top = vals[vals >= top_thr]
    return {
        'grad_mean': float(vals.mean()) if vals.size else 0.0,
        'grad_std': float(vals.std()) if vals.size else 0.0,
        'grad_max': float(vals.max()) if vals.size else 0.0,
        'grad_p95': float(np.quantile(vals, 0.95)) if vals.size else 0.0,
        'grad_top20_mean': float(top.mean()) if top.size else 0.0,
        'sobel_x_abs_mean': float(np.abs(gx[mask]).mean()) if mask.any() else 0.0,
        'sobel_y_abs_mean': float(np.abs(gy[mask]).mean()) if mask.any() else 0.0,
    }


def wavelet_features(patch: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    img = np.where(mask, patch, 0.0).astype(np.float32)
    coeffs = pywt.wavedec2(img, wavelet=WAVELET, level=WAVELET_LEVEL, mode='periodization')
    cA2 = coeffs[0]
    cH2, cV2, cD2 = coeffs[1]
    cH1, cV1, cD1 = coeffs[2]
    raw = {
        'wave_LL2_energy': float(np.mean(cA2 ** 2)),
        'wave_LH2_energy': float(np.mean(cH2 ** 2)),
        'wave_HL2_energy': float(np.mean(cV2 ** 2)),
        'wave_HH2_energy': float(np.mean(cD2 ** 2)),
        'wave_LH1_energy': float(np.mean(cH1 ** 2)),
        'wave_HL1_energy': float(np.mean(cV1 ** 2)),
        'wave_HH1_energy': float(np.mean(cD1 ** 2)),
    }
    low = raw['wave_LL2_energy']
    mid = raw['wave_LH2_energy'] + raw['wave_HL2_energy'] + raw['wave_HH2_energy']
    high = raw['wave_LH1_energy'] + raw['wave_HL1_energy'] + raw['wave_HH1_energy']
    total = low + mid + high + 1e-12
    raw.update({
        'wave_low_prop': float(low / total),
        'wave_mid_prop': float(mid / total),
        'wave_high_prop': float(high / total),
        'wave_detail_prop': float((mid + high) / total),
        'wave_high_low_ratio': float(high / (low + 1e-12)),
    })
    return raw


def fft_local_features(patch: np.ndarray, mask: np.ndarray, bins: int = 12) -> dict[str, float]:
    img = np.where(mask, patch, 0.0).astype(np.float32)
    img = img - float(img.mean())
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    rr = rr / (rr.max() + 1e-6)
    ring = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        vals = mag[(rr >= lo) & (rr < hi)]
        ring.append(float(vals.mean()) if vals.size else 0.0)
    ring = np.asarray(ring, dtype=np.float64)
    ring = ring / (ring.sum() + 1e-12)
    x = np.arange(bins, dtype=np.float64)
    centroid = float((x * ring).sum() / (ring.sum() + 1e-12))
    bandwidth = float(np.sqrt((((x - centroid) ** 2) * ring).sum() / (ring.sum() + 1e-12)))
    out = {
        'fft_low': float(ring[:4].sum()),
        'fft_mid': float(ring[4:8].sum()),
        'fft_high': float(ring[8:].sum()),
        'fft_high_low_ratio': float(ring[8:].sum() / (ring[:4].sum() + 1e-12)),
        'fft_centroid': centroid,
        'fft_bandwidth': bandwidth,
    }
    for i, v in enumerate(ring, start=1):
        out[f'fft_ring_{i:02d}'] = float(v)
    return out


def glcm_entropy_features(patch: np.ndarray, mask: np.ndarray, levels: int = 16) -> dict[str, float]:
    q = np.clip(np.floor(patch * levels), 0, levels - 1).astype(np.int32)
    valid = mask.astype(bool)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in offsets:
        y0a, y1a = max(0, -dy), q.shape[0] - max(0, dy)
        x0a, x1a = max(0, -dx), q.shape[1] - max(0, dx)
        a = q[y0a:y1a, x0a:x1a]
        b = q[y0a + dy:y1a + dy, x0a + dx:x1a + dx]
        m = valid[y0a:y1a, x0a:x1a] & valid[y0a + dy:y1a + dy, x0a + dx:x1a + dx]
        if np.any(m):
            idx = a[m] * levels + b[m]
            glcm += np.bincount(idx, minlength=levels * levels).reshape(levels, levels)
    p = glcm / (glcm.sum() + 1e-12)
    ii, jj = np.indices((levels, levels))
    contrast = float(((ii - jj) ** 2 * p).sum())
    homogeneity = float((p / (1.0 + np.abs(ii - jj))).sum())
    energy = float(np.sqrt((p ** 2).sum()))
    entropy_glcm = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    vals = patch[mask]
    return {
        'entropy_hist': safe_entropy(vals),
        'glcm_contrast': contrast,
        'glcm_homogeneity': homogeneity,
        'glcm_energy': energy,
        'glcm_entropy': entropy_glcm,
    }


def color_family_features(patch: np.ndarray, mask: np.ndarray, lut: np.ndarray) -> dict[str, float]:
    rgb = apply_lut(patch, lut)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    _, _, grad = gradient_maps(patch)
    base = mask & (sat >= 0.12) & (val >= 0.08)
    weights = np.where(base, 0.1 + grad / (grad[mask].max() + 1e-6 if mask.any() else 1.0), 0.0)
    total_area = float(mask.sum() + 1e-12)
    total_weight = float(weights.sum() + 1e-12)
    fam_masks = {
        'vermelho_rosado': base & ((hue < 18) | (hue >= 300)),
        'roxo_azulado': base & (hue >= 195) & (hue < 300),
        'esverdeado': base & (hue >= 75) & (hue < 195),
        'amarelado_marrom': base & (hue >= 18) & (hue < 75),
    }
    out: dict[str, float] = {}
    for name, fm in fam_masks.items():
        out[f'cor_{name}_area'] = float(fm.sum() / total_area)
        out[f'cor_{name}_grad'] = float(weights[fm].sum() / total_weight)
    return out


def iter_patches(gray: np.ndarray, mask: np.ndarray):
    h, w = gray.shape
    coords = []
    for y in range(0, h - PATCH_SIZE + 1, PATCH_STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_STRIDE):
            pm = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if pm.mean() >= 0.12:
                coords.append((y, x))
    if not coords:
        coords = [(max(0, (h - PATCH_SIZE) // 2), max(0, (w - PATCH_SIZE) // 2))]
    for y, x in coords:
        yield y, x, gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE], mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]


def extract_patch_features(patch: np.ndarray, pmask: np.ndarray, lut: np.ndarray) -> dict[str, float]:
    if not pmask.any():
        pmask = np.ones_like(patch, dtype=bool)
    feats: dict[str, float] = {
        'patch_mean': float(patch[pmask].mean()),
        'patch_std': float(patch[pmask].std()),
        'patch_p95': float(np.quantile(patch[pmask], 0.95)),
        'patch_mask_frac': float(pmask.mean()),
    }
    feats.update(wavelet_features(patch, pmask))
    feats.update(gradient_features(patch, pmask))
    feats.update(color_family_features(patch, pmask, lut))
    feats.update(glcm_entropy_features(patch, pmask))
    feats.update(fft_local_features(patch, pmask))
    feats['selection_score'] = float(
        feats['grad_p95'] + feats['wave_detail_prop'] + feats['fft_high'] + 0.15 * feats['entropy_hist']
    )
    return feats


def aggregate_image_features(row: pd.Series, lut: np.ndarray) -> dict[str, object]:
    img = read_rsna_image(Path(row['image_path']), 'mammography')
    gray = normalize_image(img)
    mask = breast_mask(gray)
    patch_rows = []
    for y, x, patch, pmask in iter_patches(gray, mask):
        feats = extract_patch_features(patch, pmask, lut)
        feats['patch_y'] = y
        feats['patch_x'] = x
        patch_rows.append(feats)
    patch_rows.sort(key=lambda d: d['selection_score'], reverse=True)
    top = patch_rows[:TOP_PATCHES]
    feat_names = [k for k in top[0].keys() if k not in {'patch_y', 'patch_x'}]
    out: dict[str, object] = {
        'split': row['split'],
        'patient_id': row.get('patient_id', ''),
        'image_id': row.get('image_id', ''),
        'image_path': row['image_path'],
        'cancer': int(row['cancer']),
        'n_patches': len(patch_rows),
        'n_top_patches': len(top),
    }
    for name in feat_names:
        vals = np.asarray([p[name] for p in top], dtype=np.float64)
        out[f'{name}_mean'] = float(vals.mean())
        out[f'{name}_std'] = float(vals.std())
        out[f'{name}_max'] = float(vals.max())
    best = top[0]
    for name in feat_names:
        out[f'top1_{name}'] = float(best[name])
    return out


def build_dataset(split: pd.DataFrame, lut: np.ndarray, max_images: int | None = None) -> pd.DataFrame:
    if max_images:
        pos = split[split['cancer'] == 1]
        neg = split[split['cancer'] == 0]
        half = max_images // 2
        split = pd.concat([
            pos.sample(min(len(pos), half), random_state=RANDOM_STATE),
            neg.sample(min(len(neg), max_images - min(len(pos), half)), random_state=RANDOM_STATE),
        ]).sort_index()
    rows = []
    total = len(split)
    for i, (_, row) in enumerate(split.iterrows(), start=1):
        try:
            rows.append(aggregate_image_features(row, lut))
        except Exception as exc:
            print(f'ERRO {row.get("image_path", "")}: {exc}')
        if i % 25 == 0 or i == total:
            print(f'Features extraidas {i}/{total}')
    return pd.DataFrame(rows)


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test, out_dir: Path) -> dict[str, object]:
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
        proba = (score - score.min()) / (score.max() - score.min() + 1e-12)
    pred = (proba >= 0.5).astype(int)
    metrics = {
        'model': name,
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'pr_auc': float(average_precision_score(y_test, proba)),
        'accuracy': float(accuracy_score(y_test, pred)),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
        'classification_report': classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{name} AUC={metrics["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Falso positivo')
    plt.ylabel('Verdadeiro positivo')
    plt.title(f'ROC - {name}')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f'roc_{name}.png', dpi=160)
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f'{name} PR={metrics["pr_auc"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precisao')
    plt.title(f'Precision-Recall - {name}')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f'pr_{name}.png', dpi=160)
    plt.close()
    pd.DataFrame({'y_true': y_test, 'prob_cancer': proba, 'pred': pred}).to_csv(out_dir / f'predicoes_{name}.csv', index=False)
    return metrics


def train_models(features: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    meta_cols = {'split', 'patient_id', 'image_id', 'image_path', 'cancer'}
    feature_cols = [c for c in features.columns if c not in meta_cols]
    train = features[features['split'] == 'train'].copy()
    test = features[features['split'] == 'test'].copy()
    X_train = train[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(np.float32)
    y_train = train['cancer'].to_numpy(np.int64)
    X_test = test[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(np.float32)
    y_test = test['cancer'].to_numpy(np.int64)

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    xgb = XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='binary:logistic',
        eval_metric='aucpr',
        scale_pos_weight=neg / pos,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    xgb.fit(X_train, y_train)
    metrics = {'feature_count': len(feature_cols), 'train_n': int(len(train)), 'test_n': int(len(test)), 'models': []}
    metrics['models'].append(evaluate_model('xgboost', xgb, X_train, y_train, X_test, y_test, out_dir))

    imp = pd.DataFrame({'feature': feature_cols, 'importance': xgb.feature_importances_}).sort_values('importance', ascending=False)
    imp.to_csv(out_dir / 'xgboost_importancia_features.csv', index=False)
    plt.figure(figsize=(9, 8))
    top = imp.head(30).iloc[::-1]
    plt.barh(top['feature'], top['importance'])
    plt.title('Top 30 features - XGBoost')
    plt.tight_layout()
    plt.savefig(out_dir / 'xgboost_top30_importancia.png', dpi=160)
    plt.close()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svm = SVC(C=2.0, gamma='scale', kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_s, y_train)
    metrics['models'].append(evaluate_model('svm_rbf', svm, X_train_s, y_train, X_test_s, y_test, out_dir))

    with (out_dir / 'modelo_xgboost.pkl').open('wb') as f:
        pickle.dump({'model': xgb, 'feature_cols': feature_cols}, f)
    with (out_dir / 'modelo_svm_rbf.pkl').open('wb') as f:
        pickle.dump({'model': svm, 'scaler': scaler, 'feature_cols': feature_cols}, f)
    (out_dir / 'metricas_modelos.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def write_report(metrics: dict[str, object], out_dir: Path) -> None:
    lines = []
    lines.append('RELATORIO - CLASSIFICADOR POR PATCHES COM WAVELET + GRADIENTE + COR + ENTROPIA + FFT\n')
    lines.append(f'Patch size: {PATCH_SIZE}, stride: {PATCH_STRIDE}, top patches: {TOP_PATCHES}\n')
    lines.append(f'Features: {metrics["feature_count"]}\n')
    lines.append(f'Treino: {metrics["train_n"]} imagens | Teste: {metrics["test_n"]} imagens\n\n')
    for model in metrics['models']:
        lines.append(f'Modelo: {model["model"]}\n')
        lines.append(f'ROC AUC: {model["roc_auc"]:.4f}\n')
        lines.append(f'PR AUC: {model["pr_auc"]:.4f}\n')
        lines.append(f'Accuracy: {model["accuracy"]:.4f}\n')
        lines.append(f'Matriz confusao [[TN, FP], [FN, TP]]: {model["confusion_matrix"]}\n')
        rep = model['classification_report']
        if '1' in rep:
            lines.append(f'Cancer precision: {rep["1"]["precision"]:.4f} | recall: {rep["1"]["recall"]:.4f} | f1: {rep["1"]["f1-score"]:.4f}\n')
        lines.append('\n')
    lines.append('Arquivos principais:\n')
    lines.append(str(out_dir / 'features_imagem_patches.csv') + '\n')
    lines.append(str(out_dir / 'xgboost_importancia_features.csv') + '\n')
    lines.append(str(out_dir / 'xgboost_top30_importancia.png') + '\n')
    lines.append(str(out_dir / 'metricas_modelos.json') + '\n')
    (out_dir / 'relatorio_classificador_patches.txt').write_text(''.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Reextrair features mesmo se CSV existir')
    parser.add_argument('--max-images', type=int, default=0, help='Opcional: limitar imagens para teste rapido')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lut = load_lut()
    feature_csv = OUT_DIR / 'features_imagem_patches.csv'
    if feature_csv.exists() and not args.force:
        features = pd.read_csv(feature_csv)
        print(f'Carreguei features existentes: {feature_csv}')
    else:
        split = pd.read_csv(SPLIT_CSV)
        features = build_dataset(split, lut, max_images=args.max_images or None)
        features.to_csv(feature_csv, index=False)
        print(f'Features salvas em: {feature_csv}')
    metrics = train_models(features, OUT_DIR)
    write_report(metrics, OUT_DIR)
    print('\nResumo modelos:')
    for model in metrics['models']:
        print(f"{model['model']}: ROC_AUC={model['roc_auc']:.4f} PR_AUC={model['pr_auc']:.4f} ACC={model['accuracy']:.4f} CM={model['confusion_matrix']}")
    print(f'\nSaida: {OUT_DIR}')


if __name__ == '__main__':
    main()
