#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import pydicom

from lidc_sistema_integrado import (
    mammography_feature_names,
    read_rsna_image,
    select_no_filter_features,
    train_torch_mlp_scores,
    best_f1_threshold,
    summarize_binary_predictions,
)


ROOT = Path("/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem")
OUT_DIR = ROOT / "sistema_integrado"
ANALYSIS_DIR = OUT_DIR / "topologia_frequencia"


def feature_indices(names, predicate):
    idx = [i for i, name in enumerate(names) if predicate(str(name))]
    if not idx:
        raise RuntimeError("Nenhuma feature encontrada para esta selecao.")
    return idx


def train_variant(label, X, y, train_idx, test_idx, names, random_state=42):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    score, device = train_torch_mlp_scores(X_train_s, y_train, X_test_s, random_state=random_state)
    threshold, _ = best_f1_threshold(y_test, score)
    pred = (score >= threshold).astype(int)
    row = {
        "comparacao": label,
        "n_features": int(X.shape[1]),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "pr_auc": float(average_precision_score(y_test, score)),
        "backend": "torch_mlp",
        "device": device,
        **summarize_binary_predictions(y_test, score, pred),
    }
    return row


def local_dominant_frequency_map(img, window=32, stride=8):
    arr = cv2.resize(img.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
    h, w = arr.shape
    rows = []
    ys = list(range(0, h - window + 1, stride))
    xs = list(range(0, w - window + 1, stride))
    yy, xx = np.indices((window, window))
    cy = cx = window // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (rr.max() + 1e-6)
    ring_edges = np.linspace(0.0, 1.0, 17)

    for y0 in ys:
        row = []
        for x0 in xs:
            patch = arr[y0:y0 + window, x0:x0 + window]
            patch = patch - patch.mean()
            mag = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
            band_energy = []
            for lo, hi in zip(ring_edges[:-1], ring_edges[1:]):
                band = mag[(rr >= lo) & (rr < hi)]
                band_energy.append(float(band.mean()) if band.size else 0.0)
            # Ignora DC/quase-DC para nao deixar o passa-baixa dominar o mapa.
            dominant = int(np.argmax(band_energy[1:]) + 1)
            row.append(dominant / 15.0)
        rows.append(row)

    coarse = np.asarray(rows, dtype=np.float32)
    dense = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    dense = np.clip(dense, 0.0, 1.0)
    return arr.astype(np.uint8), dense


def pixelwise_dominant_frequency_map_cuda(img, window=31, out_size=(263, 241), batch_size=65536):
    """Mapa de frequência dominante calculado para todos os pixels com PyTorch/CUDA.

    Cada pixel recebe uma janela centrada nele. A FFT 2D da janela é calculada em
    lotes na GPU quando CUDA está disponível. O retorno é um mapa denso pixel a
    pixel, sem interpolar uma grade grossa.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = cv2.resize(img.astype(np.float32), (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)
    arr = (arr - float(arr.min())) / (float(arr.max() - arr.min()) + 1e-6)
    h, w = arr.shape
    if window % 2 == 0:
        window += 1
    pad = window // 2

    t = torch.from_numpy(arr)[None, None].to(device=device, dtype=torch.float32)
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    patches = F.unfold(t, kernel_size=window, stride=1).squeeze(0).transpose(0, 1)
    n_pixels = patches.shape[0]

    yy, xx = torch.meshgrid(
        torch.arange(window, device=device),
        torch.arange(window, device=device),
        indexing="ij",
    )
    cy = cx = window // 2
    rr = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    rr = rr / (rr.max() + 1e-6)
    ring_masks = []
    edges = torch.linspace(0.0, 1.0, 17, device=device)
    for i in range(16):
        ring_masks.append(((rr >= edges[i]) & (rr < edges[i + 1])).float().reshape(-1))
    ring_masks = torch.stack(ring_masks, dim=0)
    ring_counts = ring_masks.sum(dim=1).clamp_min(1.0)

    dominant = torch.empty(n_pixels, device=device, dtype=torch.float32)
    for start in range(0, n_pixels, batch_size):
        batch = patches[start:start + batch_size].reshape(-1, window, window)
        batch = batch - batch.mean(dim=(1, 2), keepdim=True)
        spec = torch.fft.fftshift(torch.fft.fft2(batch), dim=(-2, -1)).abs()
        flat = spec.reshape(spec.shape[0], -1)
        energy = flat @ ring_masks.T
        energy = energy / ring_counts
        # Ignora DC/quase-DC para nao deixar o passa-baixa dominar.
        idx = torch.argmax(energy[:, 1:], dim=1).float() + 1.0
        dominant[start:start + batch.shape[0]] = idx / 15.0

    dense = dominant.reshape(h, w).detach().cpu().numpy()
    return (arr * 255).astype(np.uint8), dense, str(device)


def topology_metrics(freq_map):
    high = (freq_map >= np.percentile(freq_map, 85)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([], dtype=np.int32)
    return high, {
        "area_alta_freq": int(high.sum()),
        "componentes_alta_freq": int(max(n_labels - 1, 0)),
        "maior_componente_alta_freq": int(areas.max()) if areas.size else 0,
        "freq_media": float(freq_map.mean()),
        "freq_p85": float(np.percentile(freq_map, 85)),
        "freq_p95": float(np.percentile(freq_map, 95)),
    }


def segment_frequency_topology(img_small, freq_map, min_area=24):
    threshold = max(float(np.percentile(freq_map, 85)), 1e-6)
    mask = (freq_map >= threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    low = cv2.GaussianBlur(img_small.astype(np.float32), (9, 9), 0)
    high = img_small.astype(np.float32) - low

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segments = []
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            mask[labels == label_id] = 0
            continue

        component = labels == label_id
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        low_energy = float(np.mean(np.abs(low[component]))) if area else 0.0
        high_energy = float(np.mean(np.abs(high[component]))) if area else 0.0
        segments.append({
            "segmento_id": int(len(segments) + 1),
            "area_px": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "centro_x": float(cx),
            "centro_y": float(cy),
            "freq_media_segmento": float(freq_map[component].mean()),
            "passa_baixa_energia_segmento": low_energy,
            "passa_alta_energia_segmento": high_energy,
            "razao_alta_baixa_segmento": float(high_energy / (low_energy + 1e-6)),
        })

    return mask.astype(np.uint8), segments


def save_frequency_panel(row, out_path, pixelwise_cuda=True):
    img = read_rsna_image(Path(row.image_path), "mammography")
    original_h, original_w = img.shape[:2]

    if pixelwise_cuda:
        img_small, freq_map, device = pixelwise_dominant_frequency_map_cuda(img)
    else:
        img_small, freq_map = local_dominant_frequency_map(img)
        device = "cpu_grid"

    low = cv2.GaussianBlur(img_small.astype(np.float32), (9, 9), 0)
    high = img_small.astype(np.float32) - low
    high_vis = cv2.normalize(np.abs(high), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _topo_mask, metrics = topology_metrics(freq_map)
    seg_mask, segments = segment_frequency_topology(img_small, freq_map)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(img_small, cmap="gray")
    axes[0].set_title(f"Original cancer={int(row.cancer)}")
    axes[1].imshow(low, cmap="gray")
    axes[1].set_title("Passa-baixa")
    axes[2].imshow(high_vis, cmap="gray")
    axes[2].set_title("Passa-alta")
    im = axes[3].imshow(freq_map, cmap="turbo", vmin=0, vmax=1)
    axes[3].set_title("Freq. dominante local")
    axes[4].imshow(img_small, cmap="gray")
    axes[4].contour(seg_mask, levels=[0.5], colors="red", linewidths=0.8)
    axes[4].set_title(f"Segmentos={len(segments)}")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    metrics["device"] = device
    metrics["altura_mapa"] = int(freq_map.shape[0])
    metrics["largura_mapa"] = int(freq_map.shape[1])
    metrics["altura_original"] = int(original_h)
    metrics["largura_original"] = int(original_w)
    metrics["n_segmentos"] = int(len(segments))
    metrics["area_segmentada_px"] = int(seg_mask.sum())

    seg_rows = []
    for seg in segments:
        seg_rows.append({
            **seg,
            "patient_id": row.patient_id,
            "image_id": row.image_id,
            "cancer": int(row.cancer),
            "altura_original": int(original_h),
            "largura_original": int(original_w),
            "altura_analisada": int(freq_map.shape[0]),
            "largura_analisada": int(freq_map.shape[1]),
            "arquivo_png": str(out_path),
        })
    return metrics, seg_rows

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(OUT_DIR / "X_features_wavelet.npy")
    y = np.load(OUT_DIR / "y_features.npy")
    meta = pd.read_csv(OUT_DIR / "amostras_features_wavelet.csv")
    split = pd.read_csv(OUT_DIR / "split_por_series_uid.csv")
    names = mammography_feature_names(use_wavelet=True)

    train_idx = split.index[split["split"] == "train"].to_numpy()
    test_idx = split.index[split["split"] == "test"].to_numpy()

    no_filter_X, no_filter_names = select_no_filter_features(X, names)
    no_low_idx = feature_indices(names, lambda n: not n.startswith("low_"))
    only_no_low_X = X[:, no_low_idx]
    high_topology_idx = feature_indices(
        names,
        lambda n: n.startswith(("high_", "gradient_", "fft_", "wavelet_")),
    )
    high_topology_X = X[:, high_topology_idx]

    rows = [
        train_variant("sem_filtros_basicos", no_filter_X, y, train_idx, test_idx, no_filter_names),
        train_variant("com_todos_menos_passa_baixa", only_no_low_X, y, train_idx, test_idx, [names[i] for i in no_low_idx]),
        train_variant("so_alta_freq_gradiente_fft_wavelet", high_topology_X, y, train_idx, test_idx, [names[i] for i in high_topology_idx]),
        train_variant("com_todos_inclui_passa_baixa", X, y, train_idx, test_idx, names),
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ANALYSIS_DIR / "comparacao_sem_passa_baixa_topologia.csv", index=False)

    test_meta = meta.iloc[test_idx].copy().reset_index(drop=True)
    positives = test_meta[test_meta["cancer"] == 1].head(3)
    negatives = test_meta[test_meta["cancer"] == 0].head(3)
    panel_rows = pd.concat([positives, negatives], ignore_index=True)
    topo_rows = []
    segment_rows = []
    for idx, row in panel_rows.iterrows():
        out_path = ANALYSIS_DIR / f"topologia_freq_{idx:02d}_cancer{int(row.cancer)}_{row.patient_id}_{row.image_id}.png"
        metrics, seg_rows = save_frequency_panel(row, out_path, pixelwise_cuda=True)
        metrics.update({
            "arquivo": str(out_path),
            "patient_id": row.patient_id,
            "image_id": row.image_id,
            "cancer": int(row.cancer),
        })
        topo_rows.append(metrics)
        segment_rows.extend(seg_rows)
    pd.DataFrame(topo_rows).to_csv(ANALYSIS_DIR / "metricas_topologia_frequencia.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(ANALYSIS_DIR / "segmentos_topologia_frequencia.csv", index=False)

    summary = {
        "comparacao": str(ANALYSIS_DIR / "comparacao_sem_passa_baixa_topologia.csv"),
        "metricas_topologia": str(ANALYSIS_DIR / "metricas_topologia_frequencia.csv"),
        "segmentos": str(ANALYSIS_DIR / "segmentos_topologia_frequencia.csv"),
        "pasta_png": str(ANALYSIS_DIR),
        "escopo": "somente processamento de imagem: filtros, frequencia dominante pixel a pixel e segmentacao topologica",
    }
    (ANALYSIS_DIR / "resumo_topologia_frequencia.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"\nResultados salvos em: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
