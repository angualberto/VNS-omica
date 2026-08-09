#!/usr/bin/env python3
"""Ferramentas de figuras cientificas para o sistema LIDC-IDRI.

As funcoes aqui sao pensadas para gerar graficos limpos e publicaveis
em artigos, com foco em contraste, legibilidade e layout consistente.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _ensure_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    denom = arr.max() + 1e-6
    arr = (255.0 * arr / denom).clip(0, 255).astype(np.uint8)
    return arr


def _resize_mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask).astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def _binary_error(mask_gt: Optional[np.ndarray], mask_pred: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask_gt is None or mask_pred is None:
        return None
    gt = (_resize_mask(mask_gt, shape) > 0.5).astype(np.uint8)
    pred = (_resize_mask(mask_pred, shape) > 0.5).astype(np.uint8)
    return (gt != pred).astype(np.uint8)


def _overlay_mask(ax, img_gray, mask, cmap, alpha, title):
    ax.imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    if mask is not None:
        ax.imshow(mask, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def gerar_figura_artigo(
    img,
    mask_gt=None,
    mask_pred=None,
    save_path="figura.png",
):
    return gerar_figuras_artigo(img, mask_gt=mask_gt, mask_pred=mask_pred, save_path=save_path)


def gerar_figuras_artigo(
    img,
    mask_gt=None,
    mask_pred=None,
    save_path="figura.png",
):
    """Gera figura principal com comparacao de intensidade, frequencia e segmentacao.

    Layout:
    - Original
    - Passa-baixa
    - Passa-alta
    - FFT
    - Wavelet LL/LH/HL/HH
    - Ground Truth
    - Prediction
    - Overlay
    - Error map
    """

    _paper_style()
    img_gray = _ensure_grayscale_uint8(img)
    low = cv2.GaussianBlur(img_gray, (5, 5), 0)
    high = cv2.Laplacian(img_gray, cv2.CV_32F)
    high_disp = np.abs(high)
    high_disp = (255.0 * high_disp / (high_disp.max() + 1e-6)).clip(0, 255).astype(np.uint8)

    fft = np.fft.fftshift(np.fft.fft2(img_gray))
    mag = np.log1p(np.abs(fft))
    mag = (255.0 * mag / (mag.max() + 1e-6)).clip(0, 255).astype(np.uint8)

    coeffs = pywt.wavedec2(img_gray.astype(np.float32), wavelet="db4", level=2, mode="periodization")
    cA2, (cH2, cV2, cD2) = coeffs[0], coeffs[1]
    wavelets = [
        (cA2, "Wavelet LL"),
        (np.abs(cH2), "Wavelet LH"),
        (np.abs(cV2), "Wavelet HL"),
        (np.abs(cD2), "Wavelet HH"),
    ]
    wavelet_vis = []
    for arr, _ in wavelets:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.abs(arr)
        arr = (255.0 * arr / (arr.max() + 1e-6)).clip(0, 255).astype(np.uint8)
        wavelet_vis.append(arr)

    mask_gt_r = _resize_mask(mask_gt, img_gray.shape) if mask_gt is not None else None
    mask_pred_r = _resize_mask(mask_pred, img_gray.shape) if mask_pred is not None else None
    error = _binary_error(mask_gt, mask_pred, img_gray.shape)

    _paper_style()
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)

    axes = axes.ravel()
    axes[0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(low, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Low-pass")
    axes[2].imshow(high_disp, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("High-pass")
    axes[3].imshow(mag, cmap="magma")
    axes[3].set_title("FFT Spectrum")

    axes[4].imshow(wavelet_vis[0], cmap="gray", vmin=0, vmax=255)
    axes[4].set_title("Wavelet LL")
    axes[5].imshow(wavelet_vis[1], cmap="gray", vmin=0, vmax=255)
    axes[5].set_title("Wavelet LH")
    axes[6].imshow(wavelet_vis[2], cmap="gray", vmin=0, vmax=255)
    axes[6].set_title("Wavelet HL")
    axes[7].imshow(wavelet_vis[3], cmap="gray", vmin=0, vmax=255)
    axes[7].set_title("Wavelet HH")

    _overlay_mask(axes[8], img_gray, mask_gt_r, "Reds", 0.45, "Ground Truth")
    _overlay_mask(axes[9], img_gray, mask_pred_r, "jet", 0.40, "Prediction")

    axes[10].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    if mask_pred_r is not None:
        axes[10].imshow(mask_pred_r, cmap="jet", alpha=0.35, vmin=0, vmax=1)
    axes[10].set_title("Overlay")
    axes[10].axis("off")

    if error is not None:
        axes[11].imshow(error, cmap="inferno", vmin=0, vmax=1)
        axes[11].set_title("Error")
    else:
        axes[11].axis("off")
    axes[11].axis("off")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def gerar_figura_poster(
    img,
    save_path="poster.png",
):
    """Versao compacta em linha para estilo poster.

    Mantem a leitura visual em quatro blocos:
    original, low-pass, high-pass e wavelet.
    """

    _paper_style()
    img_gray = _ensure_grayscale_uint8(img)
    low = cv2.GaussianBlur(img_gray, (5, 5), 0)
    high = img_gray.astype(np.float32) - low.astype(np.float32)
    high_vis = np.abs(high)
    high_vis = (255.0 * high_vis / (high_vis.max() + 1e-6)).clip(0, 255).astype(np.uint8)

    coeffs = pywt.wavedec2(img_gray.astype(np.float32), wavelet="db4", level=2, mode="periodization")
    cA2, (cH2, cV2, cD2) = coeffs[0], coeffs[1]
    wavelet_blocks = [
        cA2,
        np.abs(cH2),
        np.abs(cV2),
        np.abs(cD2),
    ]
    wavelet_panel = np.hstack([
        (255.0 * b / (np.max(b) + 1e-6)).clip(0, 255).astype(np.uint8) for b in wavelet_blocks
    ])

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    panels = [
        (img_gray, "Original", "gray"),
        (low, "Low-pass", "gray"),
        (high_vis, "High-pass", "gray"),
        (wavelet_panel, "Wavelet (LL|LH|HL|HH)", "gray"),
    ]
    for ax, (panel, title, cmap) in zip(axes, panels):
        ax.imshow(panel, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metricas(y_true, y_prob, save_path="metricas.png"):
    """Plota ROC e Precision-Recall no mesmo arquivo."""

    _paper_style()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].plot(fpr, tpr, color="navy", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(rec, prec, color="darkred", lw=2, label=f"PR AUC = {pr_auc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].grid(alpha=0.2)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def plot_histograma_intensidades(
    img,
    mask_roi=None,
    save_path="histograma_features.png",
    bins=64,
):
    """Compara distribuicao de intensidades entre ROI e fundo."""

    _paper_style()
    img_gray = _ensure_grayscale_uint8(img)
    if mask_roi is None:
        raise ValueError("mask_roi e obrigatoria para o histograma ROI vs fundo.")

    mask = _resize_mask(mask_roi, img_gray.shape) > 0.5
    roi_vals = img_gray[mask]
    bg_vals = img_gray[~mask]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    ax.hist(bg_vals.ravel(), bins=bins, density=True, alpha=0.55, color="#4c72b0", label="Fundo")
    ax.hist(roi_vals.ravel(), bins=bins, density=True, alpha=0.55, color="#c44e52", label="ROI")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_title("Intensity Histogram: ROI vs Background")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def gerar_exemplos_segmentacao(
    imagens: Iterable[np.ndarray],
    masks_gt: Optional[Iterable[np.ndarray]] = None,
    masks_pred: Optional[Iterable[np.ndarray]] = None,
    save_path="exemplos_segmentacao.png",
    nmax: int = 8,
):
    """Gera grid com varios exemplos de segmentacao."""

    _paper_style()
    imgs = list(imagens)[:nmax]
    gts = list(masks_gt)[:nmax] if masks_gt is not None else [None] * len(imgs)
    preds = list(masks_pred)[:nmax] if masks_pred is not None else [None] * len(imgs)

    n = len(imgs)
    if n == 0:
        raise ValueError("Nenhuma imagem fornecida.")

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img = _ensure_grayscale_uint8(imgs[i])
        gt = _resize_mask(gts[i], img.shape) if gts[i] is not None else None
        pred = _resize_mask(preds[i], img.shape) if preds[i] is not None else None

        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(gt if gt is not None else np.zeros_like(img), cmap="Reds", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred if pred is not None else np.zeros_like(img), cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction")
        axes[i, 3].imshow(img, cmap="gray", vmin=0, vmax=255)
        if pred is not None:
            axes[i, 3].imshow(pred, cmap="jet", alpha=0.35, vmin=0, vmax=1)
        axes[i, 3].set_title("Overlay")
        for j in range(4):
            axes[i, j].axis("off")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
