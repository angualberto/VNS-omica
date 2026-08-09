#!/usr/bin/env python3
"""
Compare two pipelines: spectral+fuzzy selection + model, and a fuzzy-neural network.
Extracts FFT, wavelet, color-spectrum and dominant frequency features, trains models,
saves logs/metrics and generates PNG plots.

Usage example:
python scripts/compare_pipelines.py --csv path/to/cases.csv --out outdir --mode both --epochs 30
CSV must contain columns: `image_path`, `label` (0/1)
"""
import argparse
import os
import math
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import pywt
import scipy.fft as fft
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

try:
    import pydicom
except Exception:
    pydicom = None

# optional import for pretrain helper
try:
    from vnn_pixelwise_project.pretrain_spectral_attention import pretrain as pretrain_spectral_attention
except Exception:
    pretrain_spectral_attention = None


IMAGE_ROOT_CANDIDATES = [
    '/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_kaggle_oficial',
    '/media/angualberto/HD500_TRABALHO/cbis_ddsm/rsna_kaggle_oficial',
    '/media/angualberto/rsna_kaggle_oficial',
]

COLORS_MAP = None

def load_colors_map(colors_csv):
    import os
    df = pd.read_csv(colors_csv)
    # normalize possible hue column
    if 'hue_medio_graus' in df.columns and 'hue_medio' not in df.columns:
        df = df.rename(columns={'hue_medio_graus': 'hue_medio'})
    # frequency columns
    freq_cols = [c for c in df.columns if c.startswith('freq_')]
    extra_cols = []
    if 'hue_medio' in df.columns:
        extra_cols.append('hue_medio')
    keys = []
    if 'image_id' in df.columns:
        keys = df['image_id'].astype(str).tolist()
    # build map by basename and image_id
    cmap = {}
    for i, row in df.iterrows():
        basename = os.path.splitext(os.path.basename(row.get('image_path', '')))[0] if 'image_path' in df.columns else ''
        vals = []
        for c in freq_cols + extra_cols:
            vals.append(float(row[c]) if pd.notna(row.get(c)) else 0.0)
        if basename:
            cmap[basename] = {'vals': np.array(vals, dtype=np.float32), 'cols': freq_cols + extra_cols}
        if 'image_id' in df.columns and pd.notna(row['image_id']):
            cmap[str(row['image_id'])] = {'vals': np.array(vals, dtype=np.float32), 'cols': freq_cols + extra_cols}
    return cmap

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from skimage import filters as sk_filters
    from skimage import feature as sk_feature
    from skimage.morphology import disk
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
try:
    import petrou
    PETROU_AVAILABLE = True
except Exception:
    PETROU_AVAILABLE = False


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    assert 'image_path' in df.columns
    if 'label' not in df.columns:
        if 'cancer' in df.columns:
            df = df.rename(columns={'cancer': 'label'})
        else:
            raise ValueError("CSV precisa ter coluna 'label' ou 'cancer'")
    return df


def resolve_image_path(path):
    if os.path.exists(path):
        return path
    if 'rsna_breast_cancer_detection' in path:
        alt = path.replace('rsna_breast_cancer_detection', 'rsna_kaggle_oficial')
        if os.path.exists(alt):
            return alt
    filename = os.path.basename(path)
    for root in IMAGE_ROOT_CANDIDATES:
        alt = os.path.join(root, 'train_images')
        if os.path.exists(alt):
            for current_root, _, files in os.walk(alt):
                if filename in files:
                    return os.path.join(current_root, filename)
    return path


def load_image(path, size=None, grayscale=False):
    path = resolve_image_path(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.dcm':
        if pydicom is None:
            raise RuntimeError('pydicom não está disponível para ler arquivos DICOM (.dcm)')
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr - np.min(arr)
        denom = np.max(arr) - np.min(arr)
        if denom > 0:
            arr = arr / denom
        if grayscale:
            img = arr
        else:
            img = np.stack([arr, arr, arr], axis=-1)
        if size is not None:
            pil = Image.fromarray((img[..., 0] * 255).astype(np.uint8))
            pil = pil.resize(size, Image.BILINEAR)
            arr = np.array(pil).astype(np.float32) / 255.0
            if grayscale:
                return arr
            return np.stack([arr, arr, arr], axis=-1)
        return img.astype(np.float32)

    im = Image.open(path)
    if grayscale:
        im = im.convert('L')
    else:
        im = im.convert('RGB')
    if size:
        im = im.resize(size, Image.BILINEAR)
    return np.array(im).astype(np.float32) / 255.0


def fft_features(img_gray):
    # img_gray: 2D
    f = fft.rfft2(img_gray)
    mag = np.abs(f)
    # summary stats
    return np.array([np.mean(mag), np.std(mag), np.percentile(mag, 90)])


def dominant_frequency(img_gray):
    f = fft.rfft2(img_gray)
    mag = np.abs(f)
    # ignore DC
    mag[0, 0] = 0
    idx = np.unravel_index(np.argmax(mag), mag.shape)
    return np.array(idx, dtype=np.float32)


def wavelet_features(img_gray, wavelet='db1', level=2):
    coeffs = pywt.wavedec2(img_gray, wavelet=wavelet, level=level)
    feats = []
    for c in coeffs:
        if isinstance(c, tuple):
            for arr in c:
                feats.append(np.mean(np.abs(arr)))
                feats.append(np.std(arr))
        else:
            feats.append(np.mean(np.abs(c)))
    return np.array(feats, dtype=np.float32)


def color_spectrum(img_rgb, bins=16):
    # img_rgb: HxWx3 in [0,1]
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(img_rgb[..., ch].ravel(), bins=bins, range=(0, 1), density=True)
        feats.extend(hist.tolist())
    return np.array(feats, dtype=np.float32)


def build_feature_vector(path, resize=(128, 128)):
    img = load_image(path, size=resize, grayscale=False)
    gray = load_image(path, size=resize, grayscale=True)
    gray = gray if gray.ndim == 2 else gray[..., 0]
    feats = []
    feats.extend(fft_features(gray).tolist())
    feats.extend(dominant_frequency(gray).tolist())
    feats.extend(wavelet_features(gray).tolist())

    # color features: prefer precomputed map
    if COLORS_MAP is not None:
        key = os.path.splitext(os.path.basename(path))[0]
        entry = COLORS_MAP.get(key)
        if entry is not None:
            feats.extend(entry['vals'].tolist())
            cols = entry.get('cols', [])
            try:
                if 'freq_verde' in cols and 'freq_azul' in cols:
                    v = float(entry['vals'][cols.index('freq_verde')])
                    b = float(entry['vals'][cols.index('freq_azul')])
                    feats.append(v / (b + 1e-6))
                    feats.append(v - b)
            except Exception:
                pass
        else:
            feats.extend(color_spectrum(img, bins=16).tolist())
    else:
        feats.extend(color_spectrum(img, bins=16).tolist())

    # Extra derived features: texture, entropy, LBP, GLCM, fractal, Tsallis
    if SKIMAGE_AVAILABLE:
        try:
            ent = sk_filters.rank.entropy((gray * 255).astype('uint8'), disk(5)).mean()
            feats.append(float(ent))
        except Exception:
            feats.append(0.0)
        try:
            lbp = sk_feature.local_binary_pattern((gray * 255).astype('uint8'), P=8, R=1, method='uniform')
            (hist, _edges) = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            feats.extend(hist.tolist())
        except Exception:
            feats.extend([0.0] * 10)
        try:
            img_q = (gray * 7).astype('uint8')
            glcm = sk_feature.greycomatrix(img_q, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
            contrast = sk_feature.greycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = sk_feature.greycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = sk_feature.greycoprops(glcm, 'homogeneity')[0, 0]
            energy = sk_feature.greycoprops(glcm, 'energy')[0, 0]
            feats.extend([float(contrast), float(dissimilarity), float(homogeneity), float(energy)])
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        try:
            vals, _ = np.histogram((gray * 255).astype('uint8').ravel(), bins=256, range=(0, 255), density=True)
            p = vals + 1e-12
            q = 1.5
            tsallis = (1 - np.sum(p ** q)) / (q - 1)
            feats.append(float(tsallis))
        except Exception:
            feats.append(0.0)
        try:
            def fractal_dim(Z):
                Z = (Z > Z.mean()).astype(int)
                sizes = 2 ** np.arange(1, int(np.log2(min(Z.shape))) - 1)
                counts = []
                for size in sizes:
                    S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0), np.arange(0, Z.shape[1], size), axis=1)
                    counts.append(np.sum(S > 0))
                if len(sizes) < 2:
                    return 0.0
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                return -coeffs[0]
            fd = fractal_dim((gray * 255).astype('uint8'))
            feats.append(float(fd))
        except Exception:
            feats.append(0.0)
    else:
        feats.append(0.0)
        feats.extend([0.0] * 10)
        feats.extend([0.0, 0.0, 0.0, 0.0])
        feats.append(0.0)
        feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def fuzzify_features(x, centers=None, sigma=0.1):
    # simple gaussian membership over provided centers per feature
    x = np.asarray(x)
    if centers is None:
        # choose 3 centers per feature: min, median, max (scaled)
        centers = np.stack([np.zeros_like(x), 0.5 * np.ones_like(x), np.ones_like(x)], axis=1)
    # centers: (F, C)
    if centers.ndim == 2:
        centers = centers
    memberships = []
    for i in range(x.shape[0]):
        c = centers[i]
        m = np.exp(-0.5 * ((x[i] - c) / sigma) ** 2)
        memberships.extend(m.tolist())
    return np.array(memberships, dtype=np.float32)


class FeatureDataset(Dataset):
    def __init__(self, df, feature_cache=None, resize=(128, 128)):
        self.df = df.reset_index(drop=True)
        self.resize = resize
        self.feature_cache = feature_cache or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        if path in self.feature_cache:
            feats = self.feature_cache[path]
        else:
            feats = build_feature_vector(path, resize=self.resize)
            self.feature_cache[path] = feats
        label = float(row['label'])
        return feats.astype(np.float32), np.array(label, dtype=np.float32)


class SimpleFuzzyMLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def compute_pos_weight(labels, device, multiplier=1.0):
    labels = np.asarray(labels).astype(int)
    pos = max(1, int(labels.sum()))
    neg = max(1, int((labels == 0).sum()))
    weight = (neg / pos) * float(multiplier)
    return torch.tensor([weight], device=device, dtype=torch.float32)


def train_model(model, loader, opt, device, pos_weight=None):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            prob = torch.sigmoid(out).cpu().numpy()
            ps.extend(prob.tolist())
            ys.extend(yb.numpy().tolist())
    y = np.array(ys)
    p = np.array(ps)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    pr = average_precision_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    acc = accuracy_score(y, (p > 0.5).astype(int))
    return {'roc_auc': float(auc), 'pr_auc': float(pr), 'accuracy': float(acc)}


def best_epoch_summary(df_metrics):
    if df_metrics.empty:
        return {}
    roc_series = pd.to_numeric(df_metrics['roc_auc'], errors='coerce')
    pr_series = pd.to_numeric(df_metrics['pr_auc'], errors='coerce')
    loss_series = pd.to_numeric(df_metrics['loss'], errors='coerce')
    valid_roc = roc_series.dropna()
    valid_pr = pr_series.dropna()
    valid_loss = loss_series.dropna()

    if valid_roc.empty:
        best_roc = df_metrics.iloc[-1]
    else:
        best_roc = df_metrics.loc[valid_roc.idxmax()]

    if valid_pr.empty:
        best_pr = df_metrics.iloc[-1]
    else:
        best_pr = df_metrics.loc[valid_pr.idxmax()]

    if valid_loss.empty:
        best_loss = df_metrics.iloc[-1]
    else:
        best_loss = df_metrics.loc[valid_loss.idxmin()]
    return {
        'best_roc_epoch': int(best_roc['epoch']),
        'best_roc_auc': float(best_roc['roc_auc']),
        'best_pr_epoch': int(best_pr['epoch']),
        'best_pr_auc': float(best_pr['pr_auc']),
        'best_loss_epoch': int(best_loss['epoch']),
        'best_loss': float(best_loss['loss']),
        'final_epoch': int(df_metrics['epoch'].iloc[-1]),
        'final_roc_auc': float(df_metrics['roc_auc'].iloc[-1]),
        'final_pr_auc': float(df_metrics['pr_auc'].iloc[-1]),
        'final_accuracy': float(df_metrics['accuracy'].iloc[-1]),
        'final_loss': float(df_metrics['loss'].iloc[-1]),
    }


def write_comparison_summary(out_dir, results, runtime_seconds):
    rows = []
    for model_name, df_metrics in results.items():
        row = {'model': model_name}
        row.update(best_epoch_summary(df_metrics))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary['runtime_seconds'] = runtime_seconds
    summary_path = os.path.join(out_dir, 'comparison_summary.csv')
    summary.to_csv(summary_path, index=False)

    if not summary.empty:
        plot_df = summary.sort_values('best_roc_auc', ascending=True)
        plt.figure(figsize=(8, 4))
        plt.barh(plot_df['model'], plot_df['best_roc_auc'], label='Best ROC AUC')
        plt.barh(plot_df['model'], plot_df['best_pr_auc'], alpha=0.6, label='Best PR AUC')
        plt.xlabel('Score')
        plt.title('Robust comparison summary')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'comparison_summary.png'), dpi=160)
        plt.close()

    with open(os.path.join(out_dir, 'comparison_summary.md'), 'w', encoding='utf-8') as handle:
        handle.write('# Comparison summary\n\n')
        if summary.empty:
            handle.write('No model results available.\n')
        else:
            handle.write(summary.to_string(index=False))
        handle.write('\n\n')
        handle.write(f'- runtime_seconds: {runtime_seconds:.2f}\n')
        handle.write(f'- models: {", ".join(summary["model"].tolist()) if not summary.empty else "none"}\n')

    return summary_path


def run_experiment(df_train, df_val, out_dir, mode, epochs=20, batch_size=32, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    cache = {}
    ds_train = FeatureDataset(df_train, feature_cache=cache)
    ds_val = FeatureDataset(df_val, feature_cache=cache)
    tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    va = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)

    results = {}

    if mode in ('fuzzy_nn', 'both'):
        print('Running fuzzy neural network...')
        sample_feat, _ = ds_train[0]
        # fuzzify once per sample inside dataset transform
        # prepare datasets with fuzzified features
        def fuzzify_ds(ds):
            X = []
            Y = []
            for i in range(len(ds)):
                f, y = ds[i]
                f = (f - f.min()) / (f.max() - f.min() + 1e-9)
                ff = fuzzify_features(f)
                X.append(ff)
                Y.append(y)
            X = np.stack(X)
            Y = np.stack(Y)
            return X, Y

        Xtr, Ytr = fuzzify_ds(ds_train)
        Xva, Yva = fuzzify_ds(ds_val)
        pos_weight = compute_pos_weight(Ytr, device)

        tr_loader = DataLoader(list(zip(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(list(zip(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float())), batch_size=batch_size)

        model = SimpleFuzzyMLP(in_dim=Xtr.shape[1], hidden=128).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = []
        for ep in range(1, epochs + 1):
            loss = train_model(model, tr_loader, opt, device, pos_weight=pos_weight)
            metrics = eval_model(model, va_loader, device)
            metrics['loss'] = loss
            metrics['epoch'] = ep
            history.append(metrics)
            print(f"[fuzzy_nn] epoch {ep} loss={loss:.4f} roc={metrics['roc_auc']:.4f} pr={metrics['pr_auc']:.4f}")
        dfh = pd.DataFrame(history)
        dfh.to_csv(os.path.join(out_dir, 'fuzzy_nn_metrics_by_epoch.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(out_dir, 'fuzzy_nn_best.pt'))
        results['fuzzy_nn'] = dfh

        # plot
        plt.figure()
        plt.plot(dfh['epoch'], dfh['roc_auc'], label='ROC AUC')
        plt.plot(dfh['epoch'], dfh['pr_auc'], label='PR AUC')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'fuzzy_nn_metrics.png'))

    if mode in ('spectral_fuzzy', 'both'):
        print('Running spectral+fuzzy pipeline (simple classifier on selected features)...')
        # naive spectral+fuzzy selection: build features and train small MLP on raw features
        # build normalized features
        def build_XY(ds):
            X = []
            Y = []
            for i in range(len(ds)):
                f, y = ds[i]
                f = (f - f.min()) / (f.max() - f.min() + 1e-9)
                X.append(f)
                Y.append(y)
            return np.stack(X), np.stack(Y)

        Xtr, Ytr = build_XY(ds_train)
        Xva, Yva = build_XY(ds_val)
        pos_weight = compute_pos_weight(Ytr, device)

        tr_loader = DataLoader(list(zip(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(list(zip(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float())), batch_size=batch_size)

        model = SimpleFuzzyMLP(in_dim=Xtr.shape[1], hidden=128).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = []
        for ep in range(1, epochs + 1):
            loss = train_model(model, tr_loader, opt, device, pos_weight=pos_weight)
            metrics = eval_model(model, va_loader, device)
            metrics['loss'] = loss
            metrics['epoch'] = ep
            history.append(metrics)
            print(f"[spectral_fuzzy] epoch {ep} loss={loss:.4f} roc={metrics['roc_auc']:.4f} pr={metrics['pr_auc']:.4f}")
        dfh = pd.DataFrame(history)
        dfh.to_csv(os.path.join(out_dir, 'spectral_fuzzy_metrics_by_epoch.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(out_dir, 'spectral_fuzzy_best.pt'))
        results['spectral_fuzzy'] = dfh

        plt.figure()
        plt.plot(dfh['epoch'], dfh['roc_auc'], label='ROC AUC')
        plt.plot(dfh['epoch'], dfh['pr_auc'], label='PR AUC')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'spectral_fuzzy_metrics.png'))

    return results


def _get_color_vector_for_row(row):
    # prefer image_id, then basename
    key = None
    if 'image_id' in row.index and pd.notna(row.get('image_id')):
        key = str(row['image_id'])
    else:
        key = os.path.splitext(os.path.basename(row['image_path']))[0]
    entry = COLORS_MAP.get(key)
    if entry is None:
        return None
    return entry['vals'] if isinstance(entry, dict) and 'vals' in entry else entry


def run_only_dict_experiment(df_train, df_val, out_dir, epochs=20, batch_size=32, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    # build X/Y from COLORS_MAP entries matched to df rows
    def build_XY_from_df(df):
        X = []
        Y = []
        for _, row in df.iterrows():
            v = _get_color_vector_for_row(row)
            if v is None:
                continue
            X.append(v)
            y = row.get('label', None)
            if y is None and 'cancer' in row.index:
                y = row['cancer']
            if y is None:
                continue
            Y.append(int(y))
        if len(Y) == 0:
            raise RuntimeError('No samples with color vectors found for only_dict mode')
        X = np.stack(X)
        Y = np.array(Y)
        return X, Y

    Xtr, Ytr = build_XY_from_df(df_train)
    Xva, Yva = build_XY_from_df(df_val)

    # normalize per-feature
    def normalize(X):
        X = X.astype(np.float32)
        mins = X.min(axis=0, keepdims=True)
        maxs = X.max(axis=0, keepdims=True)
        return (X - mins) / (maxs - mins + 1e-9)

    Xtr = normalize(Xtr)
    Xva = normalize(Xva)
    pos_weight = compute_pos_weight(Ytr, device)

    tr_loader = DataLoader(list(zip(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(list(zip(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float())), batch_size=batch_size)

    model = SimpleFuzzyMLP(in_dim=Xtr.shape[1], hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for ep in range(1, epochs + 1):
        loss = train_model(model, tr_loader, opt, device, pos_weight=pos_weight)
        metrics = eval_model(model, va_loader, device)
        metrics['loss'] = loss
        metrics['epoch'] = ep
        history.append(metrics)
        print(f"[only_dict] epoch {ep} loss={loss:.4f} roc={metrics['roc_auc']:.4f} pr={metrics['pr_auc']:.4f}")
    dfh = pd.DataFrame(history)
    dfh.to_csv(os.path.join(out_dir, 'only_dict_metrics_by_epoch.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, 'only_dict_best.pt'))
    results = {'only_dict': dfh}
    plt.figure()
    plt.plot(dfh['epoch'], dfh['roc_auc'], label='ROC AUC')
    plt.plot(dfh['epoch'], dfh['pr_auc'], label='PR AUC')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'only_dict_metrics.png'))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--colors-csv', help='Optional precomputed colors CSV to use instead of computing color spectrum')
    parser.add_argument('--pretrain-spectral-attention', action='store_true', help='Run a quick pretrain of the spectral attention model before experiments')
    parser.add_argument('--pretrain-epochs', type=int, default=3)
    parser.add_argument('--pretrain-out-model', default=None, help='Path to save pretrained spectral attention model')
    parser.add_argument('--pretrain-batch-size', type=int, default=64)
    parser.add_argument('--mode', choices=['spectral_fuzzy', 'fuzzy_nn', 'both', 'only_dict'], default='both')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resize', type=int, nargs=2, default=(128, 128))
    parser.add_argument('--val-frac', type=float, default=0.2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    df = load_csv(args.csv)
    global COLORS_MAP
    if getattr(args, 'colors_csv', None):
        try:
            COLORS_MAP = load_colors_map(args.colors_csv)
            print('Loaded colors map with', len(COLORS_MAP), 'entries')
        except Exception as e:
            print('Failed to load colors-csv:', e)

    # Optional pretrain step
    if args.pretrain_spectral_attention:
        if pretrain_spectral_attention is None:
            print('Pretrain module not available (missing file). Skipping pretrain.')
        else:
            if not args.pretrain_out_model:
                args.pretrain_out_model = os.path.join(args.out, 'spectral_attention_pretrained.pt')
            if not getattr(args, 'colors_csv', None):
                print('--pretrain-spectral-attention requires --colors-csv; skipping')
            else:
                print('Running quick pretrain of spectral attention...')
                try:
                    pretrain_spectral_attention(args.colors_csv, args.csv, args.pretrain_out_model, epochs=args.pretrain_epochs, batch_size=args.pretrain_batch_size, device=args.device)
                except Exception as e:
                    print('Pretrain failed:', e)
    # simple split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    nval = int(len(df) * args.val_frac)
    df_val = df.iloc[:nval]
    df_train = df.iloc[nval:]

    start = time.time()
    # If only_dict mode, run specialized experiment that uses only color-dictionary vectors
    if args.mode == 'only_dict':
        if COLORS_MAP is None:
            print('Error: --mode only_dict requires --colors-csv to be provided')
            return
        results = run_only_dict_experiment(df_train, df_val, args.out, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    else:
        results = run_experiment(df_train, df_val, args.out, args.mode, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    write_comparison_summary(args.out, results, time.time() - start)
    print('Done in', time.time() - start)


if __name__ == '__main__':
    main()
