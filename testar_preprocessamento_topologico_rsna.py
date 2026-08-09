#!/usr/bin/env python3
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from analisar_topologia_frequencia_rsna import (
    pixelwise_dominant_frequency_map_cuda,
    segment_frequency_topology,
)
from lidc_sistema_integrado import (
    best_f1_threshold,
    mammography_feature_names,
    read_rsna_image,
    select_no_filter_features,
    summarize_binary_predictions,
    train_torch_mlp_scores,
)


ROOT = Path("/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem")
OUT_DIR = ROOT / "sistema_integrado"
TOPO_DIR = OUT_DIR / "topologia_frequencia"
CACHE_X = TOPO_DIR / "X_topologia_pixelwise_263x241.npy"
CACHE_NAMES = TOPO_DIR / "topologia_feature_names.txt"
RESULT_CSV = TOPO_DIR / "teste_preprocessamento_topologico.csv"


TOPOLOGY_FEATURE_NAMES = [
    "topo_freq_media",
    "topo_freq_std",
    "topo_freq_p50",
    "topo_freq_p85",
    "topo_freq_p95",
    "topo_area_segmentada_ratio",
    "topo_n_segmentos",
    "topo_maior_segmento_ratio",
    "topo_area_media_segmento",
    "topo_freq_media_segmentos",
    "topo_passa_alta_media_segmentos",
    "topo_passa_baixa_media_segmentos",
    "topo_razao_alta_baixa_media_segmentos",
]


def topology_feature_row(image_path, window=15, out_size=(128, 128)):
    img = read_rsna_image(Path(image_path), "mammography")
    img_small, freq_map, _device = pixelwise_dominant_frequency_map_cuda(
        img,
        window=window,
        out_size=out_size,
        batch_size=32768,
    )
    seg_mask, segments = segment_frequency_topology(img_small, freq_map)
    total_px = float(freq_map.size)
    if segments:
        areas = np.array([s["area_px"] for s in segments], dtype=np.float32)
        seg_freq = np.array([s["freq_media_segmento"] for s in segments], dtype=np.float32)
        seg_high = np.array([s["passa_alta_energia_segmento"] for s in segments], dtype=np.float32)
        seg_low = np.array([s["passa_baixa_energia_segmento"] for s in segments], dtype=np.float32)
        seg_ratio = np.array([s["razao_alta_baixa_segmento"] for s in segments], dtype=np.float32)
    else:
        areas = seg_freq = seg_high = seg_low = seg_ratio = np.array([0.0], dtype=np.float32)

    return np.array([
        float(freq_map.mean()),
        float(freq_map.std()),
        float(np.percentile(freq_map, 50)),
        float(np.percentile(freq_map, 85)),
        float(np.percentile(freq_map, 95)),
        float(seg_mask.sum() / (total_px + 1e-6)),
        float(len(segments)),
        float(areas.max() / (total_px + 1e-6)),
        float(areas.mean()),
        float(seg_freq.mean()),
        float(seg_high.mean()),
        float(seg_low.mean()),
        float(seg_ratio.mean()),
    ], dtype=np.float32)


def build_or_load_topology_features(meta, cache_x, cache_names, window=15, out_size=(128, 128)):
    TOPO_DIR.mkdir(parents=True, exist_ok=True)
    if cache_x.exists():
        X_topo = np.load(cache_x)
        if X_topo.shape[0] == len(meta):
            return X_topo
        print("Cache topologico tem tamanho diferente; recalculando.")

    rows = []
    for i, row in enumerate(meta.itertuples(index=False), start=1):
        rows.append(topology_feature_row(row.image_path, window=window, out_size=out_size))
        if i % 25 == 0 or i == len(meta):
            print(f"Topologia pixelwise CUDA: {i}/{len(meta)}", flush=True)
    X_topo = np.vstack(rows).astype(np.float32)
    np.save(cache_x, X_topo)
    cache_names.write_text("\n".join(TOPOLOGY_FEATURE_NAMES), encoding="utf-8")
    return X_topo


def select_without_low(X, names):
    idx = [i for i, name in enumerate(names) if not str(name).startswith("low_")]
    return X[:, idx], [names[i] for i in idx]


def select_high_topology_base(X, names):
    idx = [
        i for i, name in enumerate(names)
        if str(name).startswith(("high_", "gradient_", "fft_", "wavelet_"))
    ]
    return X[:, idx], [names[i] for i in idx]


def train_eval(label, X, y, train_idx, test_idx, random_state=42):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    score, device = train_torch_mlp_scores(X_train_s, y_train, X_test_s, random_state=random_state)
    threshold, _ = best_f1_threshold(y_test, score)
    pred = (score >= threshold).astype(int)
    return {
        "comparacao": label,
        "n_features": int(X.shape[1]),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "pr_auc": float(average_precision_score(y_test, score)),
        "backend": "torch_mlp",
        "device": device,
        **summarize_binary_predictions(y_test, score, pred),
    }


def make_pilot_subset(meta, y, split, max_samples):
    if not max_samples or max_samples <= 0 or max_samples >= len(meta):
        idx = np.arange(len(meta))
        return meta, y, split, idx

    rng = np.random.default_rng(42)
    selected = []
    for split_name in ["train", "test"]:
        split_idx = np.flatnonzero(split["split"].to_numpy() == split_name)
        per_split = max_samples // 2
        pos = split_idx[y[split_idx] == 1]
        neg = split_idx[y[split_idx] == 0]
        n_pos = min(len(pos), max(1, per_split // 2))
        n_neg = min(len(neg), per_split - n_pos)
        if n_pos:
            selected.extend(rng.choice(pos, size=n_pos, replace=False).tolist())
        if n_neg:
            selected.extend(rng.choice(neg, size=n_neg, replace=False).tolist())

    selected = np.array(sorted(set(selected)), dtype=int)
    return (
        meta.iloc[selected].reset_index(drop=True),
        y[selected],
        split.iloc[selected].reset_index(drop=True),
        selected,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--topology-height", type=int, default=128)
    parser.add_argument("--topology-width", type=int, default=128)
    parser.add_argument("--window", type=int, default=15)
    args = parser.parse_args()

    X = np.load(OUT_DIR / "X_features_wavelet.npy")
    y = np.load(OUT_DIR / "y_features.npy")
    meta = pd.read_csv(OUT_DIR / "amostras_features_wavelet.csv")
    split = pd.read_csv(OUT_DIR / "split_por_series_uid.csv")
    names = mammography_feature_names(use_wavelet=True)

    meta, y, split, selected_idx = make_pilot_subset(meta, y, split, args.max_samples)
    X = X[selected_idx]

    train_idx = split.index[split["split"] == "train"].to_numpy()
    test_idx = split.index[split["split"] == "test"].to_numpy()

    suffix = f"{len(meta)}_{args.topology_height}x{args.topology_width}_w{args.window}"
    cache_x = TOPO_DIR / f"X_topologia_pixelwise_{suffix}.npy"
    cache_names = TOPO_DIR / f"topologia_feature_names_{suffix}.txt"
    result_csv = TOPO_DIR / f"teste_preprocessamento_topologico_{suffix}.csv"

    print(f"Amostras usadas: {len(meta)} | treino={len(train_idx)} teste={len(test_idx)}", flush=True)
    print(f"Topologia pixelwise: {args.topology_height}x{args.topology_width}, janela={args.window}", flush=True)

    X_topo = build_or_load_topology_features(
        meta,
        cache_x,
        cache_names,
        window=args.window,
        out_size=(args.topology_height, args.topology_width),
    )
    X_no_filter, _ = select_no_filter_features(X, names)
    X_no_low, _ = select_without_low(X, names)
    X_high, _ = select_high_topology_base(X, names)

    rows = [
        train_eval("baseline_sem_filtros", X_no_filter, y, train_idx, test_idx),
        train_eval("melhor_anterior_sem_passa_baixa", X_no_low, y, train_idx, test_idx),
        train_eval("topologia_pixelwise_somente", X_topo, y, train_idx, test_idx),
        train_eval("sem_passa_baixa_mais_topologia", np.hstack([X_no_low, X_topo]), y, train_idx, test_idx),
        train_eval("alta_freq_mais_topologia", np.hstack([X_high, X_topo]), y, train_idx, test_idx),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(result_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nResultado salvo em: {result_csv}")


if __name__ == "__main__":
    main()
