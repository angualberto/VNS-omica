#!/usr/bin/env python3
# Sistema integrado LIDC-IDRI:
# 1. DICOM TCIA
# 2. Mascaras XML
# 3. U-Net para segmentacao
# 4. Fuzzy + Fourier + textura
# 5. Classificador final NODULO vs SEM_NODULO

import argparse
import io
import json
import random
import re
import shutil
import datetime
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import pydicom
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib.pyplot as plt

from lidc_figuras_artigo import (
    gerar_exemplos_segmentacao,
    gerar_figura_artigo,
    gerar_figuras_artigo,
    gerar_figura_poster,
    plot_histograma_intensidades,
    plot_metricas,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


DEFAULT_DATA_ROOT = Path("/media/angualberto/C8C814BEC814AD26/TCIA/LIDC-IDRI")
CLASS_NAMES = ["SEM_NODULO", "NODULO"]
DATASET_TYPES = {"auto", "lidc_ct", "mammography", "generic_dicom"}
VARIANT_LABELS = {
    "sem_wavelet": "sem_wavelet",
    "com_wavelet": "com_wavelet",
    "wavelet_unet": "wavelet_unet",
}
FILTER_FEATURE_PREFIXES = ("low_", "high_", "gradient_", "fft_", "wavelet_")
RSNA_DATASET_DIR = Path("/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_breast_cancer_detection")


def _ensure_models_dir(out_dir: Path) -> Path:
    models_dir = Path(out_dir) / "modelos"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _copy_model_to_models_dir(src_path: Path, label: str, out_dir: Path) -> Path | None:
    """Copia um artefato de modelo para `out_dir/models/` com timestamp e label.

    Retorna o path destino ou None se src_path nao existir.
    """
    src = Path(src_path)
    if not src.exists():
        return None
    models_dir = _ensure_models_dir(out_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest_name = f"{ts}_{label}{src.suffix}"
    dest = models_dir / dest_name
    shutil.copy2(src, dest)
    return dest


def prepare_output_dirs(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    dirs = {
        "root": out_dir,
        "modelos": out_dir / "modelos",
        "csv": out_dir / "csv",
        "figuras": out_dir / "figuras_artigo",
        "overlays": out_dir / "overlays",
        "relatorios": out_dir / "relatorios",
        "logs": out_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv_compat(df: pd.DataFrame, root_path: Path, csv_path: Path):
    df.to_csv(root_path, index=False)
    df.to_csv(csv_path, index=False)


def write_text_compat(text: str, root_path: Path, report_path: Path):
    root_path.write_text(text, encoding="utf-8")
    report_path.write_text(text, encoding="utf-8")


def strip_ns(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text(node, name):
    for child in node:
        if strip_ns(child.tag) == name:
            return child.text
    return None


def parse_lidc_rois(xml_zip_path, cache_path=None):
    if cache_path and Path(cache_path).exists():
        raw = json.loads(Path(cache_path).read_text())
        return {k: v for k, v in raw.items()}

    if not Path(xml_zip_path).exists():
        print(f"XML LIDC nao encontrado: {xml_zip_path}")
        print("Continuando sem anotacoes; inferencia salva predicoes, mas nao calcula metricas com ground truth.")
        return {}

    rois_by_sop = defaultdict(list)
    with zipfile.ZipFile(xml_zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        for name in tqdm(names, desc="Lendo XML"):
            try:
                root = ET.fromstring(zf.read(name))
            except ET.ParseError:
                continue

            series_uid = None
            for node in root.iter():
                if strip_ns(node.tag) == "SeriesInstanceUid":
                    series_uid = node.text
                    break

            for nodule in root.iter():
                if strip_ns(nodule.tag) != "unblindedReadNodule":
                    continue

                malignancy = None
                for child in nodule:
                    if strip_ns(child.tag) == "characteristics":
                        value = child_text(child, "malignancy")
                        malignancy = int(value) if value and value.isdigit() else None

                for roi in nodule:
                    if strip_ns(roi.tag) != "roi":
                        continue
                    if str(child_text(roi, "inclusion")).upper() != "TRUE":
                        continue
                    sop_uid = child_text(roi, "imageSOP_UID")
                    if not sop_uid:
                        continue

                    points = []
                    for edge in roi:
                        if strip_ns(edge.tag) != "edgeMap":
                            continue
                        x = child_text(edge, "xCoord")
                        y = child_text(edge, "yCoord")
                        if x is not None and y is not None:
                            points.append([int(float(x)), int(float(y))])

                    if len(points) >= 3:
                        rois_by_sop[sop_uid].append({
                            "series_uid": series_uid,
                            "points": points,
                            "malignancy": malignancy,
                        })

    rois_by_sop = dict(rois_by_sop)
    if cache_path:
        Path(cache_path).write_text(json.dumps(rois_by_sop))
    return rois_by_sop


def dicom_to_hu(ds):
    # CT armazena atenuação relativa. Para comparar pulmão entre scanners,
    # primeiro convertemos pixels para HU usando slope/intercept do DICOM.
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def hu_to_uint8_lung(hu, window_min=-1000, window_max=400):
    hu = np.clip(hu, window_min, window_max)
    return ((hu - window_min) / (window_max - window_min) * 255.0).astype(np.uint8)


def percentile_uint8(img):
    arr = img.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1 + 1e-6)
    return (arr * 255).astype(np.uint8)


def detectar_dataset_type(ds):
    modality = str(getattr(ds, "Modality", "")).upper()
    body = str(getattr(ds, "BodyPartExamined", "")).upper()
    series_desc = str(getattr(ds, "SeriesDescription", "")).upper()
    study_desc = str(getattr(ds, "StudyDescription", "")).upper()
    protocol = str(getattr(ds, "ProtocolName", "")).upper()
    text = " ".join([body, series_desc, study_desc, protocol])

    if modality in {"MG", "DX", "CR"} or "BREAST" in text or "MAMMO" in text or "MAMM" in text:
        return "mammography"
    if modality == "CT":
        return "lidc_ct"
    return "generic_dicom"


def detect_dataset_type_from_dicom(ds):
    return detectar_dataset_type(ds)


def preprocessing_name(dataset_type):
    if dataset_type == "lidc_ct":
        return "HU lung window"
    if dataset_type == "mammography":
        return "percentile normalization"
    return "percentile normalization"


def pipeline_name(dataset_type):
    if dataset_type == "lidc_ct":
        return "HU + janela pulmonar + frequência + gradiente + textura + U-Net quando houver máscara"
    if dataset_type == "mammography":
        return "frequência + gradiente + textura"
    return "normalização por percentis + frequência + gradiente + textura"


def should_use_unet(dataset_type):
    return dataset_type == "lidc_ct"


def dicom_to_uint8(ds, dataset_type):
    """Pre-processamento adaptativo por domínio.

    CT precisa de HU porque o valor físico do voxel depende de RescaleSlope e
    RescaleIntercept. Mamografia não usa HU: a intensidade é detector/fornecedor
    dependente, então percentis são mais robustos para bancos diferentes.
    """
    if dataset_type == "lidc_ct":
        return hu_to_uint8_lung(dicom_to_hu(ds))
    img = percentile_uint8(ds.pixel_array)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = 255 - img
    return img


def preprocess_dicom_image(ds, dataset_type):
    return dicom_to_uint8(ds, dataset_type)


def detect_dataset_type(series_root_dir, requested="auto", max_scan=30):
    if requested != "auto":
        return requested
    root = Path(series_root_dir)
    if (root / "train.csv").exists() and (root / "train_images").exists():
        return "mammography"
    root_text = str(series_root_dir).upper()
    if "BREAST" in root_text or "MAMMO" in root_text or "MAMM" in root_text:
        return "mammography"
    counts = Counter()
    for idx, (source_path, member) in enumerate(iter_dicoms(series_root_dir, max_series=None)):
        if idx >= max_scan:
            break
        try:
            ds = read_dicom_from_source(source_path, member, stop_before_pixels=True)
            counts[detect_dataset_type_from_dicom(ds)] += 1
        except Exception:
            continue
    if counts.get("mammography", 0) > 0:
        return "mammography"
    return counts.most_common(1)[0][0] if counts else "generic_dicom"


def resolve_rsna_paths(series_root_dir):
    root = Path(series_root_dir)
    candidates = [root, root.parent]
    for base in candidates:
        train_csv = base / "train.csv"
        images_dir = base / "train_images"
        if train_csv.exists() and images_dir.exists():
            return train_csv, images_dir
    return None, None


def is_rsna_dataset(series_root_dir):
    train_csv, images_dir = resolve_rsna_paths(series_root_dir)
    return train_csv is not None and images_dir is not None


def mask_from_rois(shape, rois):
    mask = np.zeros(shape, dtype=np.uint8)
    for roi in rois:
        pts = np.array(roi["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def crop_centered(img, mask, center_x, center_y, size):
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    x1 = max(0, x2 - size)
    y1 = max(0, y2 - size)
    crop_img = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    if crop_img.shape[0] != size or crop_img.shape[1] != size:
        crop_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return crop_img, crop_mask


def roi_center(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask.shape[1] // 2, mask.shape[0] // 2
    return int(xs.mean()), int(ys.mean())


def random_negative_patch(img8, positive_mask, size, tries=80):
    candidate = img8 > 5
    h, w = img8.shape
    half = size // 2
    ys, xs = np.where(candidate & (positive_mask == 0))
    if len(xs) == 0:
        return None
    empty = np.zeros_like(img8, dtype=np.uint8)
    for _ in range(tries):
        idx = random.randrange(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        if cx < half or cy < half or cx >= w - half or cy >= h - half:
            continue
        if positive_mask[cy-half:cy+half, cx-half:cx+half].sum() == 0:
            patch, patch_mask = crop_centered(img8, empty, cx, cy, size)
            return patch, patch_mask, cx, cy
    return None


def triangular_membership(x, a, b, c):
    left = (x - a) / (b - a + 1e-6)
    right = (c - x) / (c - b + 1e-6)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapezoid_membership(x, a, b, c, d):
    rise = (x - a) / (b - a + 1e-6)
    fall = (d - x) / (d - c + 1e-6)
    return np.clip(np.minimum(np.minimum(rise, 1.0), fall), 0.0, 1.0)


def spatial_features(patch, mask=None):
    if mask is not None and mask.sum() > 0:
        values = patch[mask > 0].astype(np.float32)
        area = float(mask.sum()) / mask.size
    else:
        values = patch.reshape(-1).astype(np.float32)
        area = 0.0
    hist, _ = np.histogram(values, bins=16, range=(0, 255), density=True)
    dark_ratio = float((values < 64).mean())
    mid_ratio = float(((values >= 64) & (values < 160)).mean())
    bright_ratio = float((values >= 160).mean())
    contrast = float(np.percentile(values, 90) - np.percentile(values, 10))
    patch_u8 = patch.astype(np.uint8)
    if mask is not None and mask.sum() > 0:
        patch_u8 = patch_u8 * mask.astype(np.uint8)
    edges = cv2.Canny(patch_u8, 40, 120)
    edge_density = float((edges > 0).mean())
    return np.array([
        values.mean(), values.std(), np.percentile(values, 10),
        np.percentile(values, 50), np.percentile(values, 90), area,
        dark_ratio, mid_ratio, bright_ratio, contrast, edge_density,
        *hist.astype(np.float32).tolist(),
    ], dtype=np.float32)


def fuzzy_tone_features(patch, mask=None):
    if mask is not None and mask.sum() > 0:
        values = patch[mask > 0].astype(np.float32)
    else:
        values = patch.reshape(-1).astype(np.float32)
    memberships = [
        trapezoid_membership(values, 0, 0, 35, 75),
        triangular_membership(values, 35, 85, 135),
        triangular_membership(values, 95, 145, 195),
        triangular_membership(values, 155, 205, 245),
        trapezoid_membership(values, 210, 240, 255, 255),
    ]
    feats = []
    for degree in memberships:
        feats.extend([float(degree.mean()), float(degree.max()), float(np.percentile(degree, 90))])
    stacked = np.vstack(memberships).T
    stacked = stacked / (stacked.sum(axis=1, keepdims=True) + 1e-6)
    entropy = -np.sum(stacked * np.log(stacked + 1e-6), axis=1)
    feats.extend([float(entropy.mean()), float(entropy.std())])
    return np.array(feats, dtype=np.float32)


def _safe_entropy(values, bins=32):
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == 0:
        return 0.0
    values = np.abs(values)
    if np.allclose(values.max(), values.min()):
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-6)
    return float(-np.sum(hist * np.log(hist + 1e-6)))


def low_high_features(patch, mask=None):
    img = patch.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        img = img * mask.astype(np.float32)
    low = cv2.GaussianBlur(img, (9, 9), 0)
    high = img - low
    low_energy = float(np.mean(low ** 2))
    high_energy = float(np.mean(high ** 2))
    mean_low = float(np.mean(low))
    std_low = float(np.std(low))
    mean_high = float(np.mean(high))
    std_high = float(np.std(high))
    low_var = float(np.var(low))
    high_var = float(np.var(high))
    ratio = float(high_energy / (low_energy + 1e-6))
    return np.array([
        low_energy, high_energy, ratio, low_var, high_var,
        mean_low, std_low, mean_high, std_high,
    ], dtype=np.float32)


# O gradiente em escala de cinza mede variações locais de intensidade.
# Em imagens médicas, regiões suspeitas frequentemente apresentam bordas,
# espículas, microcalcificações ou transições abruptas entre tecido normal
# e tecido alterado. Por isso, features de gradiente complementam Wavelet,
# Fourier e filtros passa-alta.
def gradient_features(img, mask=None):
    """
    Extrai features de gradiente em escala de cinza.
    O gradiente mede variacoes locais de intensidade, destacando bordas,
    transicoes bruscas, espiculas, microestruturas e irregularidades.
    """
    arr = img.astype(np.float32)

    if mask is not None and mask.sum() > 0:
        arr = arr * mask.astype(np.float32)

    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    _ang = np.arctan2(gy, gx)

    values = mag[mask > 0] if mask is not None and mask.sum() > 0 else mag.ravel()
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    hist, _ = np.histogram(values, bins=16, range=(0, values.max() + 1e-6), density=True)
    edge_threshold = float(values.mean() + values.std())
    edge_density = float((values > edge_threshold).mean())

    return np.array([
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
        float(values.max()),
        float((values > np.percentile(values, 90)).mean()),
        edge_density,
        *hist.astype(np.float32).tolist(),
    ], dtype=np.float32)


def texture_uniformity_features(img, mask=None):
    arr = img.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        values = arr[mask > 0]
    else:
        values = arr.ravel()
    if values.size == 0:
        return np.array([0.0], dtype=np.float32)
    texture_uniformity = float(1.0 / (1.0 + np.var(values)))
    return np.array([texture_uniformity], dtype=np.float32)


def fft_features(patch, mask=None, bins=16):
    img = patch.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        img = img * mask.astype(np.float32)
    img = img - img.mean()
    fft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(fft))
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (rr.max() + 1e-6)

    ring_feats = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        band = mag[(rr >= lo) & (rr < hi)]
        ring_feats.append(float(band.mean()) if band.size else 0.0)
    ring_feats = np.array(ring_feats, dtype=np.float32)
    ring_feats = ring_feats / (ring_feats.sum() + 1e-6)

    low = float(ring_feats[: bins // 3].sum())
    mid = float(ring_feats[bins // 3 : 2 * bins // 3].sum())
    high = float(ring_feats[2 * bins // 3 :].sum())
    centroid = float((np.arange(bins, dtype=np.float32) * ring_feats).sum() / (ring_feats.sum() + 1e-6))
    bandwidth = float(np.sqrt(((np.arange(bins, dtype=np.float32) - centroid) ** 2 * ring_feats).sum() / (ring_feats.sum() + 1e-6)))
    total_energy = float(ring_feats.sum())
    fft_low_energy = low
    fft_mid_energy = mid
    fft_high_energy = high
    high_low_ratio = float(high / (low + 1e-6))
    return np.array([
        total_energy,
        centroid,
        bandwidth,
        fft_low_energy,
        fft_mid_energy,
        fft_high_energy,
        high_low_ratio,
        *ring_feats.tolist(),
    ], dtype=np.float32)


def wavelet_features(patch, mask=None):
    img = patch.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        img = img * mask.astype(np.float32)
    coeffs = pywt.wavedec2(img, wavelet="db4", level=2, mode="periodization")
    cA2 = coeffs[0]
    cH2, cV2, cD2 = coeffs[1]
    bands = {"LL": cA2, "LH": cH2, "HL": cV2, "HH": cD2}
    feats = []
    energy_low = float(np.sum(cA2 ** 2))
    energy_high = float(np.sum(cH2 ** 2) + np.sum(cV2 ** 2) + np.sum(cD2 ** 2))
    for name in ("LL", "LH", "HL", "HH"):
        band = np.asarray(bands[name], dtype=np.float32)
        feats.extend([
            float(np.sum(band ** 2)),
            float(band.mean()),
            float(band.std()),
            _safe_entropy(band),
        ])
    feats.extend([
        float(energy_high / (energy_low + 1e-6)),
        float(energy_low),
        float(energy_high),
    ])
    return np.array(feats, dtype=np.float32)


def extract_features(patch, mask=None, use_wavelet=True):
    parts = [
        spatial_features(patch, mask),
        fuzzy_tone_features(patch, mask),
        low_high_features(patch, mask),
        gradient_features(patch, mask),
        fft_features(patch, mask),
        unet_morph_features(patch, mask),
        advanced_shape_spectral_features(patch, mask),
    ]
    names = feature_names(use_wavelet=use_wavelet, include_unet=True)
    if use_wavelet:
        parts.insert(5, wavelet_features(patch, mask))
    return np.concatenate(parts), names


def extract_features_ct(img, mask=None, use_wavelet=True):
    return extract_features(img, mask, use_wavelet=use_wavelet)


def extract_features_mammo(img, mask=None, use_wavelet=True):
    # Mamografia não usa HU nem morfologia de U-Net. O sinal físico principal é
    # intensidade normalizada, textura, bordas, espículas e conteúdo frequencial.
    parts = [
        spatial_features(img, mask),
        low_high_features(img, mask),
        gradient_features(img, mask),
        fft_features(img, mask),
    ]
    if use_wavelet:
        parts.append(wavelet_features(img, mask))
    parts.extend([
        fuzzy_tone_features(img, mask),
        texture_uniformity_features(img, mask),
    ])
    names = mammography_feature_names(use_wavelet=use_wavelet)
    return np.concatenate(parts), names


def extract_features_generic(img, mask=None, use_wavelet=True):
    parts = [
        spatial_features(img, mask),
        low_high_features(img, mask),
        gradient_features(img, mask),
        fft_features(img, mask),
    ]
    if use_wavelet:
        parts.append(wavelet_features(img, mask))
    parts.extend([
        fuzzy_tone_features(img, mask),
        texture_uniformity_features(img, mask),
    ])
    names = mammography_feature_names(use_wavelet=use_wavelet)
    return np.concatenate(parts), names


def extract_features_by_domain(img, dataset_type, mask=None, use_wavelet=True):
    if dataset_type == "lidc_ct":
        return extract_features_ct(img, mask, use_wavelet=use_wavelet)
    if dataset_type == "mammography":
        return extract_features_mammo(img, mask, use_wavelet=use_wavelet)
    return extract_features_generic(img, mask, use_wavelet=use_wavelet)


def extract_features_no_wavelet(patch, mask=None):
    return extract_features(patch, mask, use_wavelet=False)


def unet_morph_features(img, mask=None):
    """Morfologia da máscara prevista.

    Mesmo que a U-Net nao seja perfeita, sua saida fornece pistas estruturais
    uteis: area, forma, numero de componentes e estatisticas dentro da lesao.
    """
    if mask is None or mask.sum() == 0:
        return np.zeros(9, dtype=np.float32)

    img = img.astype(np.float32)
    mask_bin = (mask > 0).astype(np.uint8)
    area = float(mask_bin.sum()) / mask_bin.size
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    num_components = max(0, num_labels - 1)
    if num_components == 0:
        return np.zeros(9, dtype=np.float32)

    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(comp_areas)) + 1
    largest_area = float(comp_areas[largest_idx - 1]) / mask_bin.size
    left = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    top = int(stats[largest_idx, cv2.CC_STAT_TOP])
    width = max(1, int(stats[largest_idx, cv2.CC_STAT_WIDTH]))
    height = max(1, int(stats[largest_idx, cv2.CC_STAT_HEIGHT]))
    bbox_ratio = float(width / height)
    largest_component = largest_area

    largest_mask = (labels == largest_idx).astype(np.uint8)
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    circularity = 0.0
    eccentricity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        raw_perimeter = float(cv2.arcLength(contour, True))
        perimeter = raw_perimeter / max(1.0, 2 * (mask_bin.shape[0] + mask_bin.shape[1]))
        area_cnt = float(cv2.contourArea(contour))
        circularity = float(4 * np.pi * area_cnt / (raw_perimeter ** 2 + 1e-6))
        if len(contour) >= 5:
            (_, _), (a, b), _ = cv2.fitEllipse(contour)
            major = max(a, b)
            minor = min(a, b)
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor ** 2) / (major ** 2 + 1e-6))))
        else:
            ys, xs = np.where(largest_mask > 0)
            if len(xs) > 5:
                cov = np.cov(np.vstack([xs, ys]))
                eigvals = np.sort(np.linalg.eigvalsh(cov))
                if eigvals[-1] > 0:
                    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[0] / (eigvals[-1] + 1e-6))))

    values = img[mask_bin > 0]
    mean_inside = float(values.mean()) if values.size else 0.0
    std_inside = float(values.std()) if values.size else 0.0
    return np.array([
        area,
        perimeter,
        circularity,
        eccentricity,
        bbox_ratio,
        largest_component,
        float(num_components),
        mean_inside,
        std_inside,
    ], dtype=np.float32)


def advanced_shape_spectral_features(patch, mask=None):
    img = patch.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        mask_bin = (mask > 0).astype(np.uint8)
        img_masked = img * mask_bin.astype(np.float32)
    else:
        mask_bin = np.zeros_like(img, dtype=np.uint8)
        img_masked = img

    low = cv2.GaussianBlur(img_masked, (5, 5), 0)
    high = img_masked - low
    texture_score = float(np.mean(high ** 2) / (np.mean(low ** 2) + 1e-6))

    solidity = 0.0
    compactness = 0.0
    if mask_bin.sum() > 0:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area_cnt = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(area_cnt / (hull_area + 1e-6))
            compactness = float((perimeter ** 2) / (area_cnt + 1e-6))

    return np.array([texture_score, solidity, compactness], dtype=np.float32)


def spatial_feature_names():
    return [
        "spatial_mean", "spatial_std", "spatial_p10", "spatial_p50", "spatial_p90",
        "spatial_roi_area", "spatial_dark_ratio", "spatial_mid_ratio",
        "spatial_bright_ratio", "spatial_contrast", "spatial_edge_density",
        *[f"spatial_hist_{i:02d}" for i in range(16)],
    ]


def fuzzy_feature_names():
    return [
        *[
            f"fuzzy_{tone}_{stat}"
            for tone in ("very_dark", "dark", "mid", "bright", "very_bright")
            for stat in ("mean", "max", "p90")
        ],
        "fuzzy_entropy_mean",
        "fuzzy_entropy_std",
    ]


def low_high_feature_names():
    return [
        "low_energy", "high_energy", "low_high_ratio", "low_var", "high_var",
        "low_mean", "low_std", "high_mean", "high_std",
    ]


def gradient_feature_names():
    return [
        "gradient_mean", "gradient_std", "gradient_p10", "gradient_p50",
        "gradient_p90", "gradient_max", "gradient_edge_ratio", "gradient_edge_density",
        *[f"gradient_hist_{i:02d}" for i in range(16)],
    ]


def fft_feature_names():
    return [
        "fft_total_energy", "fft_centroid", "fft_bandwidth", "fft_low_energy",
        "fft_mid_energy", "fft_high_energy", "fft_high_low_ratio",
        *[f"fft_ring_{i:02d}" for i in range(16)],
    ]


def wavelet_feature_names():
    return [
        *[
            f"wavelet_{band}_{stat}"
            for band in ("LL", "LH", "HL", "HH")
            for stat in ("energy", "mean", "std", "entropy")
        ],
        "wavelet_high_low_ratio",
        "wavelet_low_energy",
        "wavelet_high_energy",
    ]


def unet_feature_names():
    return [
        "mask_area", "mask_perimeter", "mask_circularity", "mask_eccentricity",
        "mask_bbox_ratio", "mask_largest_component", "mask_num_components",
        "mask_mean_inside", "mask_std_inside", "texture_score", "mask_solidity",
        "mask_compactness",
    ]


def unet_prediction_feature_names():
    return [
        "unet_area", "unet_perimeter", "unet_circularity", "unet_eccentricity",
        "unet_bbox_ratio", "unet_largest_component", "unet_num_components",
        "unet_mean_inside", "unet_std_inside", "unet_prob_mean",
    ]


FEATURE_REGISTRY = {
    "spatial": spatial_feature_names(),
    "fuzzy": fuzzy_feature_names(),
    "low_high": low_high_feature_names(),
    "gradient": gradient_feature_names(),
    "fft": fft_feature_names(),
    "wavelet": wavelet_feature_names(),
    "unet_morph": unet_feature_names(),
}


def feature_names(use_wavelet=True, include_unet=False):
    names = []
    names += FEATURE_REGISTRY["spatial"]
    names += FEATURE_REGISTRY["fuzzy"]
    names += FEATURE_REGISTRY["low_high"]
    names += FEATURE_REGISTRY["gradient"]
    names += FEATURE_REGISTRY["fft"]
    if use_wavelet:
        names += FEATURE_REGISTRY["wavelet"]
    if include_unet:
        names += FEATURE_REGISTRY["unet_morph"]
    return names


def feature_names_for_domain(dataset_type, variant_key):
    if dataset_type in {"mammography", "generic_dicom"}:
        if variant_key == "sem_wavelet":
            return mammography_feature_names(use_wavelet=False)
        return mammography_feature_names(use_wavelet=True)
    return feature_names_for_variant(variant_key)


def feature_names_for_variant(variant_key):
    if variant_key == "sem_wavelet":
        return feature_names(use_wavelet=False)
    if variant_key == "com_wavelet":
        return feature_names(use_wavelet=True)
    if variant_key == "wavelet_unet":
        return feature_names(use_wavelet=True) + unet_prediction_feature_names()
    raise ValueError(f"Versao de features desconhecida: {variant_key}")


def mammography_feature_names(use_wavelet=True):
    names = []
    names += FEATURE_REGISTRY["spatial"]
    names += FEATURE_REGISTRY["low_high"]
    names += FEATURE_REGISTRY["gradient"]
    names += FEATURE_REGISTRY["fft"]
    if use_wavelet:
        names += FEATURE_REGISTRY["wavelet"]
    names += FEATURE_REGISTRY["fuzzy"]
    names += ["texture_uniformity"]
    return names


def align_feature_matrix_by_name(X, current_names, expected_names, fill_value=0.0, warn=True):
    X = np.asarray(X, dtype=np.float32)
    current_index = {name: idx for idx, name in enumerate(current_names)}
    aligned = np.full((X.shape[0], len(expected_names)), fill_value, dtype=np.float32)
    missing = []
    for out_idx, name in enumerate(expected_names):
        src_idx = current_index.get(name)
        if src_idx is None:
            missing.append(name)
            continue
        aligned[:, out_idx] = X[:, src_idx]
    extra = [name for name in current_names if name not in set(expected_names)]
    if warn and missing:
        print(f"Aviso: {len(missing)} features esperadas nao existem na extracao atual; preenchidas com 0.")
    if warn and extra:
        print(f"Aviso: {len(extra)} features novas ignoradas porque nao existem no modelo treinado.")
    return aligned


def legacy_feature_names(use_wavelet=True):
    """Nomes dos modelos antigos, antes de gradiente e features avançadas.

    Esses nomes preservam a ordem real em que os modelos legados foram treinados
    (espacial, fuzzy, Fourier, Wavelet opcional, low/high-pass e máscara).
    Assim a inferência continua alinhando por nome, sem cortar colunas por posição.
    """
    names = []
    names += [
        "spatial_mean", "spatial_std", "spatial_p10", "spatial_p50", "spatial_p90",
        "spatial_roi_area", "spatial_dark_ratio", "spatial_mid_ratio",
        "spatial_bright_ratio", "spatial_contrast", "spatial_edge_density",
    ]
    names += [f"spatial_hist_{i:02d}" for i in range(16)]
    names += [
        f"fuzzy_{tone}_{stat}"
        for tone in ("very_dark", "dark", "mid", "bright", "very_bright")
        for stat in ("mean", "max", "p90")
    ]
    names += ["fuzzy_entropy_mean", "fuzzy_entropy_std"]
    names += [
        "fft_total_energy", "fft_centroid", "fft_bandwidth", "fft_low_energy",
        "fft_mid_energy", "fft_high_energy", "fft_high_low_ratio",
    ]
    names += [f"fft_ring_{i:02d}" for i in range(16)]
    if use_wavelet:
        names += [
            f"wavelet_{band}_{stat}"
            for band in ("LL", "LH", "HL", "HH")
            for stat in ("energy", "mean", "std", "entropy")
        ]
        names += ["wavelet_high_low_ratio", "wavelet_low_energy", "wavelet_high_energy"]
    names += ["low_energy", "high_energy", "low_high_ratio", "low_var", "high_var"]
    names += [
        "mask_area", "mask_perimeter", "mask_circularity", "mask_eccentricity",
        "mask_bbox_ratio", "mask_largest_component", "mask_num_components",
        "mask_mean_inside", "mask_std_inside",
    ]
    return names


def legacy_feature_names_for_variant(variant_key, expected_count):
    if variant_key == "sem_wavelet" and expected_count == 81:
        return legacy_feature_names(use_wavelet=False)
    if variant_key == "com_wavelet" and expected_count == 100:
        return legacy_feature_names(use_wavelet=True)
    if variant_key == "wavelet_unet" and expected_count == 110:
        return legacy_feature_names(use_wavelet=True) + unet_prediction_feature_names()
    return None

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.b = DoubleConv(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(F.max_pool2d(x1, 2))
        x3 = self.d3(F.max_pool2d(x2, 2))
        xb = self.b(F.max_pool2d(x3, 2))
        y = self.u3(xb)
        y = self.c3(torch.cat([y, x3], dim=1))
        y = self.u2(y)
        y = self.c2(torch.cat([y, x2], dim=1))
        y = self.u1(y)
        y = self.c1(torch.cat([y, x1], dim=1))
        return self.out(y)


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def dice_score(logits, targets, thresh=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > thresh).float()
    num = 2 * (preds * targets).sum(dim=(2, 3)) + eps
    den = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (num / den).mean().item()


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """
    Focal Loss concentra gradiente em pixels dificeis.
    Isso e adequado para nodulos porque a mascara positiva costuma ocupar uma
    area pequena em relacao ao fundo.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()


class UNetDataset(Dataset):
    def __init__(self, items, rois_by_sop, img_size=256, augment=False, dataset_type="lidc_ct"):
        self.items = items
        self.rois_by_sop = rois_by_sop
        self.img_size = img_size
        self.dataset_type = dataset_type
        # Sem augmentation por requisito cientifico do experimento.
        # A melhora da U-Net vem da funcao de perda, nao de flip/rotacao/ruido.
        self.augment = False

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        zip_path, member, sop_uid = self.items[idx]
        ds = read_dicom_from_source(zip_path, member)
        img = dicom_to_uint8(ds, self.dataset_type)
        mask = mask_from_rois(img.shape, self.rois_by_sop[sop_uid])
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy((img.astype(np.float32) / 255.0)[None]), torch.from_numpy(mask.astype(np.float32)[None])


def iter_dicoms(series_root_dir, max_series=None):
    root = Path(series_root_dir)
    zip_paths = sorted(root.glob("*.zip"), key=natural_path_key)
    if zip_paths:
        if max_series is not None:
            zip_paths = zip_paths[:max_series]
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path) as zf:
                members = []
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    if name.lower().endswith(".dcm"):
                        members.append(name)
                    else:
                        try:
                            with zf.open(name) as fh:
                                head = fh.read(132)
                            if len(head) >= 132 and head[128:132] == b"DICM":
                                members.append(name)
                        except Exception:
                            continue
                for member in sorted(members, key=natural_path_key):
                    yield zip_path, member
        return

    dcm_files = sorted([p for p in root.rglob("*") if p.is_file() and is_probably_dicom_file(p)], key=natural_path_key)
    series_dirs = sorted({p.parent for p in dcm_files}, key=natural_path_key)
    if max_series is not None:
        series_dirs = series_dirs[:max_series]
    for series_dir in series_dirs:
        series_dcm_files = sorted(list(series_dir.glob("*.dcm")) + list(series_dir.glob("*.DCM")), key=natural_path_key)
        for dcm_file in series_dcm_files:
            yield series_dir, str(dcm_file.relative_to(series_dir))


def natural_path_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def is_probably_dicom_file(path):
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with path.open("rb") as fh:
            head = fh.read(132)
        return len(head) >= 132 and head[128:132] == b"DICM"
    except Exception:
        return False


def count_dicom_series_dirs(series_root_dir):
    root = Path(series_root_dir)
    if not root.exists():
        return 0
    dcm_files = sorted([p for p in root.rglob("*") if p.is_file() and is_probably_dicom_file(p)], key=natural_path_key)
    return len({p.parent for p in dcm_files})


def read_dicom_from_source(source_path, member, stop_before_pixels=False):
    source_path = Path(source_path)
    if source_path.is_dir():
        dcm_path = source_path / member
        return pydicom.dcmread(str(dcm_path), force=True, stop_before_pixels=stop_before_pixels)
    with zipfile.ZipFile(source_path) as zf:
        with zf.open(member) as fh:
            return pydicom.dcmread(io.BytesIO(fh.read()), force=True, stop_before_pixels=stop_before_pixels)


def build_unet_items(series_zip_dir, rois_by_sop, max_series=None):
    items = []
    for zip_path, member in tqdm(list(iter_dicoms(series_zip_dir, max_series)), desc="Indexando U-Net"):
        ds = read_dicom_from_source(zip_path, member, stop_before_pixels=True)
        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        if sop_uid in rois_by_sop:
            items.append((str(zip_path), member, sop_uid))
    return items


def split_unet_items_by_series(items, rois_by_sop, test_size=0.2, random_state=42):
    """Split da U-Net por series_uid, mantendo series inteiras fora da validacao."""
    groups = np.array([
        str(rois_by_sop[sop_uid][0].get("series_uid", sop_uid))
        for _, _, sop_uid in items
    ])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(items, groups=groups))
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Data leakage na U-Net: {len(overlap)} series em treino e validacao.")
    print("Split U-Net por series_uid:")
    print(f"  series treino: {len(train_groups)}")
    print(f"  series validacao: {len(val_groups)}")
    return [items[i] for i in train_idx], [items[i] for i in val_idx]


def build_feature_dataset(series_zip_dir, rois_by_sop, max_series, patch_size, neg_prob, use_wavelet=True, dataset_type="lidc_ct"):
    X, y, rows = [], [], []
    feature_names = None
    for zip_path, member in tqdm(list(iter_dicoms(series_zip_dir, max_series)), desc="Extraindo features"):
        ds = read_dicom_from_source(zip_path, member)
        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
        img = preprocess_dicom_image(ds, dataset_type)
        rois = rois_by_sop.get(sop_uid, [])
        if rois:
            mask = mask_from_rois(img.shape, rois)
            cx, cy = roi_center(mask)
            patch, patch_mask = crop_centered(img, mask, cx, cy, patch_size)
            feats, names = extract_features_by_domain(patch, dataset_type, patch_mask, use_wavelet=use_wavelet)
            X.append(feats)
            feature_names = names if feature_names is None else feature_names
            y.append(1)
            rows.append({"label": "NODULO", "zip": str(zip_path), "member": member, "series_uid": series_uid, "sop_uid": sop_uid, "center_x": cx, "center_y": cy})
        if random.random() < neg_prob:
            pos_mask = mask_from_rois(img.shape, rois) if rois else np.zeros_like(img, dtype=np.uint8)
            neg = random_negative_patch(img, pos_mask, patch_size)
            if neg is not None:
                patch, patch_mask, cx, cy = neg
                feats, names = extract_features_by_domain(patch, dataset_type, patch_mask, use_wavelet=use_wavelet)
                X.append(feats)
                feature_names = names if feature_names is None else feature_names
                y.append(0)
                rows.append({"label": "SEM_NODULO", "zip": str(zip_path), "member": member, "series_uid": series_uid, "sop_uid": sop_uid, "center_x": cx, "center_y": cy})
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64), pd.DataFrame(rows), feature_names or []


def select_rsna_rows(train_csv, images_dir, max_patients=None, random_state=42):
    df = pd.read_csv(train_csv)
    required = {"patient_id", "image_id", "cancer"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"train.csv RSNA sem colunas obrigatorias: {sorted(missing)}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["cancer"] = df["cancer"].astype(int)
    df["image_path"] = df.apply(
        lambda r: str(Path(images_dir) / r["patient_id"] / f"{r['image_id']}.dcm"),
        axis=1,
    )
    df = df[df["image_path"].map(lambda p: Path(p).exists())].copy()
    if df.empty:
        raise RuntimeError(f"Nenhuma imagem RSNA encontrada em {images_dir}")

    if max_patients is not None and max_patients > 0:
        rng = np.random.default_rng(random_state)
        patient_labels = df.groupby("patient_id")["cancer"].max()
        pos_patients = np.array(sorted(patient_labels[patient_labels == 1].index))
        neg_patients = np.array(sorted(patient_labels[patient_labels == 0].index))
        n_pos = min(len(pos_patients), max(1, max_patients // 2))
        n_neg = min(len(neg_patients), max_patients - n_pos)
        chosen_pos = rng.choice(pos_patients, size=n_pos, replace=False) if n_pos else np.array([], dtype=str)
        chosen_neg = rng.choice(neg_patients, size=n_neg, replace=False) if n_neg else np.array([], dtype=str)
        chosen = set(chosen_pos.tolist() + chosen_neg.tolist())
        df = df[df["patient_id"].isin(chosen)].copy()

    df = df.sort_values(["patient_id", "image_id"]).reset_index(drop=True)
    if df["cancer"].nunique() < 2:
        counts = df["cancer"].value_counts().to_dict()
        raise RuntimeError(f"Amostra RSNA ficou com uma classe so: {counts}. Aumente --max-series.")
    return df


def read_rsna_image(image_path, dataset_type="mammography"):
    ds = pydicom.dcmread(str(image_path), force=True)
    return preprocess_dicom_image(ds, dataset_type)


def build_rsna_feature_dataset(series_root_dir, max_patients, patch_size, use_wavelet=True, dataset_type="mammography", random_state=42):
    train_csv, images_dir = resolve_rsna_paths(series_root_dir)
    if train_csv is None:
        raise RuntimeError(
            "Base RSNA nao encontrada. Informe --dicom-root apontando para a pasta que contem "
            "train.csv e train_images."
        )

    df = select_rsna_rows(train_csv, images_dir, max_patients=max_patients, random_state=random_state)
    X, y, rows = [], [], []
    feature_names = None
    for r in tqdm(list(df.itertuples(index=False)), desc="Extraindo features RSNA"):
        image_path = Path(r.image_path)
        img = read_rsna_image(image_path, dataset_type)
        patch = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        patch_mask = np.zeros_like(patch, dtype=np.uint8)
        feats, names = extract_features_by_domain(patch, dataset_type, patch_mask, use_wavelet=use_wavelet)
        label = int(r.cancer)
        X.append(feats)
        feature_names = names if feature_names is None else feature_names
        y.append(label)
        rows.append({
            "label": "NODULO" if label else "SEM_NODULO",
            "zip": str(image_path),
            "member": image_path.name,
            "series_uid": str(r.patient_id),
            "sop_uid": str(r.image_id),
            "center_x": int(img.shape[1] // 2),
            "center_y": int(img.shape[0] // 2),
            "patient_id": str(r.patient_id),
            "image_id": str(r.image_id),
            "laterality": getattr(r, "laterality", ""),
            "view": getattr(r, "view", ""),
            "cancer": label,
            "image_path": str(image_path),
        })

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64), pd.DataFrame(rows), feature_names or []


def build_full_image_dataset(series_root_dir, rois_by_sop, patch_size, max_series=None, use_wavelet=True, dataset_type="lidc_ct"):
    """
    Varre todas as fatias uma por uma.

    Para slices com ROI, o patch usa o centro da mascara real.
    Para slices sem ROI, o patch usa o centro da imagem.
    Isso evita amostragem negativa e permite inferencia/avaliacao em todas as imagens.
    """
    X, y, rows = [], [], []
    feature_names = None
    for source_path, member in tqdm(list(iter_dicoms(series_root_dir, max_series)), desc="Varredura completa"):
        ds = read_dicom_from_source(source_path, member)
        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
        img = preprocess_dicom_image(ds, dataset_type)
        rois = rois_by_sop.get(sop_uid, [])
        if rois:
            gt_mask = mask_from_rois(img.shape, rois)
            cx, cy = roi_center(gt_mask)
            patch_mask = gt_mask
            label = 1
            label_name = "NODULO"
        else:
            gt_mask = np.zeros_like(img, dtype=np.uint8)
            cx, cy = img.shape[1] // 2, img.shape[0] // 2
            patch_mask = gt_mask
            label = 0
            label_name = "SEM_NODULO"
        patch, patch_mask = crop_centered(img, patch_mask, cx, cy, patch_size)
        feats, names = extract_features_by_domain(patch, dataset_type, patch_mask, use_wavelet=use_wavelet)
        X.append(feats)
        feature_names = names if feature_names is None else feature_names
        y.append(label)
        rows.append({
            "label": label_name,
            "zip": str(source_path),
            "member": member,
            "series_uid": series_uid,
            "sop_uid": sop_uid,
            "center_x": cx,
            "center_y": cy,
        })
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64), pd.DataFrame(rows), feature_names or []


def load_patch_from_meta_row(row, rois_by_sop, series_zip_dir, patch_size, dataset_type="lidc_ct"):
    """Recupera o mesmo patch usado nas features tabulares, junto com GT e metadados."""
    image_path_value = getattr(row, "image_path", None)
    if image_path_value and not pd.isna(image_path_value):
        img = read_rsna_image(Path(image_path_value), dataset_type)
        patch = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        gt_mask = np.zeros_like(patch, dtype=np.uint8)
        return patch, gt_mask, img, np.zeros_like(img, dtype=np.uint8)

    zip_path = Path(row.zip)
    if not zip_path.is_absolute():
        zip_path = series_zip_dir / row.zip
    ds = read_dicom_from_source(zip_path, row.member)

    img = preprocess_dicom_image(ds, dataset_type)
    rois = rois_by_sop.get(str(row.sop_uid), [])
    gt_mask = mask_from_rois(img.shape, rois) if rois else np.zeros_like(img, dtype=np.uint8)
    cx = int(getattr(row, "center_x", img.shape[1] // 2))
    cy = int(getattr(row, "center_y", img.shape[0] // 2))
    patch, patch_mask = crop_centered(img, gt_mask, cx, cy, patch_size)
    return patch, patch_mask, img, gt_mask


def save_overlay_image(patch, gt_mask, pred_mask, out_path, alpha=0.45):
    # patch: uint8 grayscale, gt_mask/pred_mask: binary uint8 same shape
    vis = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    # GT em verde
    overlay[gt_mask > 0] = (0, 255, 0)
    # Pred em vermelho (pred tem prioridade visual)
    overlay[pred_mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def save_series_overlay(out_row, rois_by_sop, series_zip_dir, out_dir, model_dir, patch_size, img_size, series_uid, dataset_type="lidc_ct"):
    """Salva uma PNG por série com overlay usando a primeira fatia disponível da série."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    row = out_row.iloc[0]
    try:
        patch, patch_mask, img_full, gt_mask_full = load_patch_from_meta_row(row, rois_by_sop, series_zip_dir, patch_size, dataset_type)
    except Exception:
        return None

    pred_mask = np.zeros_like(patch, dtype=np.uint8)
    unet_path = Path(model_dir) / "unet_lidc_best.pt"
    if unet_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            unet_model = UNetSmall().to(device)
            unet_model.load_state_dict(torch.load(unet_path, map_location=device))
            unet_model.eval()
            p_resized = cv2.resize(patch, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy((p_resized.astype(np.float32) / 255.0)[None, None]).to(device)
            with torch.no_grad():
                logits = unet_model(t)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            probs_small = cv2.resize(probs, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred_mask = (probs_small > 0.5).astype(np.uint8)
        except Exception:
            pred_mask = np.zeros_like(patch, dtype=np.uint8)

    out_path = target_dir / f"overlay_series_{series_uid}.png"
    save_overlay_image(patch, patch_mask, pred_mask, out_path)
    return out_path


def train_unet(args, rois_by_sop, out_dir):
    items = build_unet_items(args.series_zip_dir, rois_by_sop, args.max_series)
    print("Fatias positivas U-Net:", len(items))
    if len(items) < 10:
        raise RuntimeError("Poucas fatias positivas para treinar U-Net.")
    train_items, val_items = split_unet_items_by_series(
        items,
        rois_by_sop,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_dl = DataLoader(UNetDataset(train_items, rois_by_sop, args.img_size, False, args.dataset_type), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(UNetDataset(val_items, rois_by_sop, args.img_size, False, args.dataset_type), batch_size=args.batch_size, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for imgs, masks in tqdm(train_dl, desc=f"U-Net treino {epoch:02d}"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = dice_loss(logits, masks) + focal_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += (dice_loss(logits, masks) + focal_loss(logits, masks)).item() * imgs.size(0)
                val_dice += dice_score(logits, masks) * imgs.size(0)
        train_loss = total / len(train_items)
        val_loss /= len(val_items)
        val_dice /= len(val_items)
        print(f"[{epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), out_dir / "unet_lidc_best.pt")
            try:
                _copy_model_to_models_dir(out_dir / "unet_lidc_best.pt", "unet_lidc_best", out_dir)
            except Exception:
                pass
    if best < 0.2:
        print("U-Net desativada: desempenho insuficiente (Dice muito baixo)")
        return best, False
    return best, True


def load_data(args, rois_by_sop, out_dir, use_wavelet=True, suffix=""):
    if args.dataset_type == "mammography" and is_rsna_dataset(args.series_zip_dir):
        X, y, meta, feature_names = build_rsna_feature_dataset(
            args.series_zip_dir,
            args.max_series,
            args.patch_size,
            use_wavelet=use_wavelet,
            dataset_type=args.dataset_type,
            random_state=args.random_state,
        )
    else:
        X, y, meta, feature_names = build_feature_dataset(
            args.series_zip_dir,
            rois_by_sop,
            args.max_series,
            args.patch_size,
            args.neg_prob,
            use_wavelet=use_wavelet,
            dataset_type=args.dataset_type,
        )
    print("X:", X.shape)
    print("Distribuicao:", {CLASS_NAMES[k]: v for k, v in Counter(y).items()})
    csv_name = "amostras_features.csv" if not suffix else f"amostras_features{suffix}.csv"
    meta.to_csv(out_dir / csv_name, index=False)
    if hasattr(args, "output_dirs"):
        meta.to_csv(args.output_dirs["csv"] / csv_name, index=False)
    return X, y, meta, feature_names


def split_data(X, y, meta, test_size=0.25, random_state=42, n_candidates=100):
    """
    Divide por grupo anatomico/estudo, nao por fatia.

    A mesma series_uid pode conter dezenas ou centenas de fatias muito parecidas.
    Se fatias da mesma serie aparecem em treino e teste, o modelo aprende
    caracteristicas daquela aquisicao/paciente e a metrica fica otimista.
    GroupShuffleSplit impede esse vazamento: cada series_uid fica inteira em
    treino OU teste.

    GroupShuffleSplit nao e estratificado por classe. Para preservar o melhor
    possivel a proporcao NODULO/SEM_NODULO, tentamos varios splits e escolhemos
    o que deixa a prevalencia do teste mais proxima da prevalencia global.
    """
    if "series_uid" not in meta.columns:
        raise ValueError("meta precisa conter a coluna 'series_uid' para split por grupo.")

    groups = meta["series_uid"].astype(str).to_numpy()
    if len(groups) != len(y):
        raise ValueError("meta e y precisam ter o mesmo numero de amostras.")

    global_pos_rate = float(np.mean(y))
    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_size,
        random_state=random_state,
    )

    best = None
    for train_idx, test_idx in splitter.split(X, y, groups):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"Data leakage detectado: {len(overlap)} series em treino e teste.")

        test_pos_rate = float(np.mean(y_test))
        train_pos_rate = float(np.mean(y_train))
        score = abs(test_pos_rate - global_pos_rate) + 0.5 * abs(train_pos_rate - global_pos_rate)

        if best is None or score < best[0]:
            best = (score, train_idx, test_idx, train_pos_rate, test_pos_rate)

    if best is None:
        raise RuntimeError("Nao foi possivel criar split por series_uid com as duas classes em treino e teste.")

    _, train_idx, test_idx, train_pos_rate, test_pos_rate = best
    print("Split por series_uid:")
    print(f"  series treino: {len(set(groups[train_idx]))}")
    print(f"  series teste:  {len(set(groups[test_idx]))}")
    print(f"  prevalencia global NODULO: {global_pos_rate:.3f}")
    print(f"  prevalencia treino NODULO: {train_pos_rate:.3f}")
    print(f"  prevalencia teste NODULO:  {test_pos_rate:.3f}")

    return train_idx, test_idx


def make_mlp_classifier(random_state=42):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        random_state=random_state,
    )


class TorchBinaryMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_torch_mlp_scores(X_train, y_train, X_test, random_state=42, epochs=220, batch_size=256):
    """Treina a MLP da comparacao em CUDA quando disponivel."""
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    model = TorchBinaryMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    generator = torch.Generator()
    generator.manual_seed(random_state)
    dataset = torch.utils.data.TensorDataset(Xtr, ytr)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(Xte)).detach().cpu().numpy()
    return score, str(device)


def get_candidate_classifiers(random_state=42):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "mlp": make_mlp_classifier(random_state),
    }
    if XGBClassifier is not None:
        xgb_kwargs = {
            "n_estimators": 350,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }
        if torch.cuda.is_available():
            xgb_kwargs.update({"tree_method": "hist", "device": "cuda"})
        models["xgboost"] = XGBClassifier(**xgb_kwargs)
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return models


def model_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def best_f1_threshold(y_true, score):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    if thresholds.size == 0:
        return 0.5, float(f1.max()) if f1.size else 0.0
    best_idx = int(np.nanargmax(f1))
    threshold_idx = min(best_idx, thresholds.size - 1)
    return float(thresholds[threshold_idx]), float(f1[best_idx])


def align_feature_matrix_for_scaler(X, scaler):
    expected = getattr(scaler, "n_features_in_", None)
    if expected is None or X.shape[1] == expected:
        return X
    if X.shape[1] > expected:
        print(
            f"Aviso: modelo salvo espera {expected} features, mas o extrator gerou {X.shape[1]}. "
            "Usando apenas as primeiras features para compatibilidade com modelo antigo."
        )
        return X[:, :expected]
    raise RuntimeError(f"Modelo espera {expected} features, mas o extrator gerou {X.shape[1]}.")


def _is_new_pipeline_pack(pack):
    return pack.get("pipeline_version") == "v2_gradient_wavelet"


def _load_model_pack_candidate(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def resolve_model_pack(model_dir, prefer_version="v2_gradient_wavelet"):
    model_dir = Path(model_dir)
    candidates = []
    root_pack_path = model_dir / "modelos_integrados_sem_randomforest.joblib"
    candidates.append(root_pack_path)
    models_dir = model_dir / "models"
    if models_dir.exists():
        candidates.extend(sorted(models_dir.glob("*.joblib"), reverse=True))

    legacy_pack = None
    for path in candidates:
        pack = _load_model_pack_candidate(path)
        if pack is None:
            continue
        if pack.get("pipeline_version") == prefer_version:
            return pack, path
        if legacy_pack is None:
            legacy_pack = (pack, path)
    if legacy_pack is not None:
        return legacy_pack
    raise FileNotFoundError(f"Nenhum pacote de modelo encontrado em {model_dir}")


def make_variant_feature_matrix(variant_key, X_no_wavelet, X_wavelet, X_unet, pack, dataset_type="lidc_ct"):
    expected_names = pack.get("feature_names")
    expected_count = pack.get("n_features_expected") or getattr(pack["scaler"], "n_features_in_", None)
    if not expected_names and expected_count is not None:
        expected_names = legacy_feature_names_for_variant(variant_key, int(expected_count))
    legacy_pack = not _is_new_pipeline_pack(pack)
    if variant_key == "sem_wavelet":
        current_names = feature_names_for_domain(dataset_type, "sem_wavelet")
        if expected_names:
            return align_feature_matrix_by_name(X_no_wavelet, current_names, expected_names, warn=not legacy_pack)
        return align_feature_matrix_for_scaler(X_no_wavelet, pack["scaler"])
    if variant_key == "com_wavelet":
        current_names = feature_names_for_domain(dataset_type, "com_wavelet")
        if expected_names:
            return align_feature_matrix_by_name(X_wavelet, current_names, expected_names, warn=not legacy_pack)
        return align_feature_matrix_for_scaler(X_wavelet, pack["scaler"])

    if variant_key == "wavelet_unet":
        current_names = feature_names_for_domain(dataset_type, "wavelet_unet")
        X_full = np.hstack([X_wavelet, X_unet])
        if expected_names:
            return align_feature_matrix_by_name(X_full, current_names, expected_names, warn=not legacy_pack)

        scaler = pack["scaler"]
        expected = getattr(scaler, "n_features_in_", None)
        if expected is None or X_full.shape[1] == expected:
            return X_full
        old_wavelet = X_wavelet[:, :-3] if X_wavelet.shape[1] >= 3 else X_wavelet
        old_full = np.hstack([old_wavelet, X_unet])
        if old_full.shape[1] == expected:
            print(
                f"Aviso: modelo salvo espera {expected} features em wavelet_unet. "
                "Removendo as 3 features novas antes de juntar U-Net para compatibilidade."
            )
            return old_full
        return align_feature_matrix_for_scaler(X_full, scaler)

    raise ValueError(f"Versao de modelo desconhecida: {variant_key}")


def summarize_binary_predictions(y_true, score, pred):
    result = {
        "total": int(len(pred)),
        "pred_nodulo": int(np.sum(pred == 1)),
        "pred_sem_nodulo": int(np.sum(pred == 0)),
        "taxa_pred_nodulo": float(np.mean(pred == 1)) if len(pred) else 0.0,
        "score_medio": float(np.mean(score)) if len(score) else 0.0,
    }
    if len(np.unique(y_true)) >= 2:
        report = classification_report(
            y_true,
            pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        result.update({
            "accuracy": float(report["accuracy"]),
            "precision_nodulo": float(report["NODULO"]["precision"]),
            "recall_nodulo": float(report["NODULO"]["recall"]),
            "f1_nodulo": float(report["NODULO"]["f1-score"]),
            "roc_auc": float(roc_auc_score(y_true, score)),
            "pr_auc": float(average_precision_score(y_true, score)),
        })
    return result


def evaluate_predictions(y_test, pred, score, out_dir, report_name, cm_name, title, threshold=0.5):
    report = classification_report(y_test, pred, target_names=CLASS_NAMES)
    roc_auc = roc_auc_score(y_test, score)
    pr_auc = average_precision_score(y_test, score)

    print(report)
    print(f"Threshold: {threshold:.6f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    (out_dir / report_name).write_text(
        report + f"\nThreshold: {threshold:.6f}\nROC AUC: {roc_auc:.4f}\nPrecision-Recall AUC: {pr_auc:.4f}\n"
    )

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / cm_name, dpi=150)
    plt.close()
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "report": report, "score": score, "threshold": threshold}


def train_best_tabular_model(X_train, y_train, X_test, y_test, random_state=42):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    for name, model in get_candidate_classifiers(random_state).items():
        model.fit(X_train_s, y_train)
        score = model_scores(model, X_test_s)
        threshold, best_f1 = best_f1_threshold(y_test, score)
        pred = (score >= threshold).astype(int)
        results[name] = {
            "model": model,
            "score": score,
            "pred": pred,
            "threshold": threshold,
            "best_f1": best_f1,
            "roc_auc": roc_auc_score(y_test, score),
            "pr_auc": average_precision_score(y_test, score),
        }
        print(
            f"{name}: ROC AUC={results[name]['roc_auc']:.4f} "
            f"PR AUC={results[name]['pr_auc']:.4f} "
            f"threshold={threshold:.6f} F1={best_f1:.4f}"
        )

    best_name = max(results, key=lambda k: (results[k]["pr_auc"], results[k]["roc_auc"]))
    print(f"Melhor threshold ({best_name}): {results[best_name]['threshold']:.6f}")
    return scaler, best_name, results[best_name]["model"], results


def select_no_filter_features(X, feature_names):
    keep_idx = [
        idx for idx, name in enumerate(feature_names)
        if not str(name).startswith(FILTER_FEATURE_PREFIXES)
    ]
    if not keep_idx:
        raise RuntimeError("Nenhuma feature sem filtros foi encontrada para comparar a rede neural.")
    return X[:, keep_idx], [feature_names[idx] for idx in keep_idx]


def select_without_wavelet_features(X, feature_names):
    keep_idx = [
        idx for idx, name in enumerate(feature_names)
        if not str(name).startswith("wavelet_")
    ]
    if not keep_idx:
        raise RuntimeError("Nenhuma feature sem Wavelet foi encontrada.")
    return X[:, keep_idx], [feature_names[idx] for idx in keep_idx]


def neural_metrics_row(label, y_true, score, pred, threshold, n_features):
    summary = summarize_binary_predictions(y_true, score, pred)
    row = {
        "comparacao": label,
        "modelo": "mlp",
        "threshold": float(threshold),
        "n_features": int(n_features),
        **summary,
    }
    return row


def train_neural_filter_comparison(
    X_no_filter_train,
    X_no_filter_test,
    X_with_filter_train,
    X_with_filter_test,
    y_train,
    y_test,
    out_dir,
    random_state=42,
):
    """Compara a mesma rede neural MLP sem filtros versus com filtros."""
    rows = []
    predictions = {"y_true": y_test}
    configs = [
        ("rede_neural_sem_filtros", X_no_filter_train, X_no_filter_test),
        ("rede_neural_com_filtros", X_with_filter_train, X_with_filter_test),
    ]

    for label, Xtr, Xte in configs:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        score, device_name = train_torch_mlp_scores(Xtr_s, y_train, Xte_s, random_state=random_state)
        threshold, _ = best_f1_threshold(y_test, score)
        pred = (score >= threshold).astype(int)
        row = neural_metrics_row(label, y_test, score, pred, threshold, Xtr.shape[1])
        row["backend"] = "torch_mlp"
        row["device"] = device_name
        rows.append(row)
        predictions[f"score_{label}"] = score
        predictions[f"pred_{label}"] = pred

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(out_dir / "comparacao_rede_neural_filtros.csv", index=False)
    pd.DataFrame(predictions).to_csv(out_dir / "predicoes_rede_neural_filtros.csv", index=False)

    lines = [
        "Comparacao da rede neural MLP para prever cancer/suspeita:",
        "Backend: PyTorch MLP com CUDA quando disponivel.",
        "",
        "- rede_neural_sem_filtros: usa apenas features basicas de intensidade/estatistica.",
        "- rede_neural_com_filtros: usa tambem filtros baixa/alta frequencia, gradiente, Fourier e Wavelet.",
        "",
        comparison_df.to_string(index=False),
        "",
    ]
    if "accuracy" in comparison_df.columns:
        best_row = comparison_df.sort_values(["accuracy", "f1_nodulo"], ascending=False).iloc[0]
        lines.append(
            f"Maior acerto: {best_row['comparacao']} "
            f"(accuracy={best_row['accuracy']:.4f}, f1_suspeito={best_row['f1_nodulo']:.4f})."
        )
    (out_dir / "relatorio_rede_neural_filtros.txt").write_text("\n".join(lines), encoding="utf-8")
    return comparison_df


def load_unet_model(out_dir, device):
    path = out_dir / "unet_lidc_best.pt"
    if not path.exists():
        path = out_dir / "modelos" / "unet_lidc_best.pt"
    if not path.exists():
        return None
    model = UNetSmall().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predicted_unet_mask_features(patch, unet_model, device, img_size=256):
    if unet_model is None:
        return np.zeros(10, dtype=np.float32)

    src_h, src_w = patch.shape
    resized = cv2.resize(patch, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = torch.from_numpy(resized[None, None]).to(device)
    with torch.no_grad():
        prob_map = torch.sigmoid(unet_model(x)).cpu().numpy()[0, 0]
    prob_patch = cv2.resize(prob_map, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    mask = (prob_patch >= 0.5).astype(np.uint8)

    area = float(mask.sum()) / mask.size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours)) / max(1.0, 2 * (src_h + src_w))
    circularity = 0.0
    eccentricity = 0.0
    bbox_ratio = 0.0
    largest = 0.0
    n_components = 0.0
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = int(np.argmax(areas))
        largest_area = float(areas[max_idx])
        largest = largest_area / mask.size
        largest_perimeter = cv2.arcLength(contours[max_idx], True)
        circularity = float(4 * np.pi * largest_area / (largest_perimeter ** 2 + 1e-6))
        x, y, w, h = cv2.boundingRect(contours[max_idx])
        bbox_ratio = float(w / (h + 1e-6))
        if len(contours[max_idx]) >= 5:
            (_, _), (a, b), _ = cv2.fitEllipse(contours[max_idx])
            major = max(a, b)
            minor = min(a, b)
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor ** 2) / (major ** 2 + 1e-6))))

    n_components = float(cv2.connectedComponents(mask)[0] - 1)

    values = patch[mask > 0].astype(np.float32)
    mean_inside = float(values.mean()) if values.size else 0.0
    std_inside = float(values.std()) if values.size else 0.0
    prob_mean = float(prob_patch.mean())
    return np.array([
        area,
        perimeter,
        circularity,
        eccentricity,
        bbox_ratio,
        largest,
        n_components,
        mean_inside,
        std_inside,
        prob_mean,
    ], dtype=np.float32)


def predict_unet_maps(patch, unet_model, device, img_size=256):
    """Retorna o mapa sigmoid e a mascara binaria da U-Net sem alterar a extracao de features."""
    if unet_model is None:
        return None, None

    src_h, src_w = patch.shape
    resized = cv2.resize(patch, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = torch.from_numpy(resized[None, None]).to(device)
    with torch.no_grad():
        prob_map = torch.sigmoid(unet_model(x)).cpu().numpy()[0, 0]
    prob_patch = cv2.resize(prob_map, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    mask = (prob_patch >= 0.5).astype(np.uint8)
    return prob_patch, mask


def build_unet_feature_matrix(meta, out_dir, patch_size, img_size, series_zip_dir, dataset_type="lidc_ct"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = load_unet_model(out_dir, device)
    if unet_model is None:
        print("U-Net treinada nao encontrada; features morfologicas serao zeros.")

    rows = []
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Features U-Net"):
        zip_path = Path(row.zip)
        if not zip_path.is_absolute():
            zip_path = series_zip_dir / row.zip
        ds = read_dicom_from_source(zip_path, row.member)
        img = preprocess_dicom_image(ds, dataset_type)
        cx = int(getattr(row, "center_x", img.shape[1] // 2))
        cy = int(getattr(row, "center_y", img.shape[0] // 2))
        patch, _ = crop_centered(img, np.zeros_like(img, dtype=np.uint8), cx, cy, patch_size)
        rows.append(predicted_unet_mask_features(patch, unet_model, device, img_size))
    return np.vstack(rows).astype(np.float32)


def plot_comparative_curves(curves, y_test, out_dir):
    for name, score in curves.items():
        RocCurveDisplay.from_predictions(y_test, score, name=name)
    plt.title("Curva ROC comparativa")
    plt.tight_layout()
    plt.savefig(out_dir / "curva_roc_comparativa.png", dpi=150)
    plt.close()

    for name, score in curves.items():
        PrecisionRecallDisplay.from_predictions(y_test, score, name=name)
    plt.title("Curva Precision-Recall comparativa")
    plt.tight_layout()
    plt.savefig(out_dir / "curva_pr_comparativa.png", dpi=150)
    plt.close()


def gerar_figura_decomposicao(img, save_path, mask=None, overlay=None):
    """Painel científico 2x5 com filtros clássicos, gradiente e Wavelet."""
    arr = img.astype(np.float32)
    low = cv2.GaussianBlur(arr, (9, 9), 0)
    high = arr - low
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(arr - arr.mean()))))
    coeffs = pywt.dwt2(arr, "db4", mode="periodization")
    ll, (lh, hl, hh) = coeffs
    if overlay is not None:
        overlay_panel = overlay
        overlay_title = "Overlay"
    elif mask is not None and np.asarray(mask).sum() > 0:
        overlay_panel = np.asarray(mask, dtype=np.float32)
        overlay_title = "Mascara"
    else:
        overlay_panel = np.zeros_like(arr, dtype=np.float32)
        overlay_title = "Overlay / mascara"
    panels = [
        ("Original", arr),
        ("Low-pass", low),
        ("High-pass", high),
        ("Gradiente Sobel", gradient),
        ("FFT Spectrum", fft),
        ("Wavelet LL", ll),
        ("Wavelet LH", lh),
        ("Wavelet HL", hl),
        ("Wavelet HH", hh),
        (overlay_title, overlay_panel),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (title, data) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_score_overlay_series(group, out_dir, threshold, series_uid):
    series = group.reset_index(drop=True)
    x = np.arange(len(series))
    y = series["score"].to_numpy(dtype=float)
    pred = series["pred"].to_numpy(dtype=int)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, color="black", linewidth=1.5, label="score")
    plt.axhline(threshold, color="tab:orange", linestyle="--", label=f"threshold={threshold:.2f}")
    if len(x):
        plt.scatter(x[pred == 1], y[pred == 1], color="red", s=24, label="suspeito")
        plt.scatter(x[pred == 0], y[pred == 0], color="tab:blue", s=14, alpha=0.6, label="baixo_score")
    plt.xlabel("Indice da fatia")
    plt.ylabel("Score")
    plt.ylim(-0.02, 1.02)
    plt.title(f"Series {str(series_uid)[:32]}...")
    plt.legend(loc="best")
    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_series_{series_uid}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def generate_article_figures(out, comparison_df, out_dirs, threshold, dataset_type, rois_by_sop, args):
    fig_dir = out_dirs["figuras"]
    if not comparison_df.empty:
        labels = comparison_df["versao"].tolist()
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x - width / 2, comparison_df["taxa_pred_nodulo"], width, label="taxa_suspeita")
        plt.bar(x + width / 2, comparison_df["score_medio"], width, label="score_medio")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "comparacao_modelos_barras.png", dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 4))
    for col in [c for c in out.columns if c.startswith("score_") and c != "score"]:
        plt.hist(out[col], bins=30, alpha=0.45, label=col.replace("score_", ""), density=True)
    plt.xlabel("Score")
    plt.ylabel("Densidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "histograma_scores.png", dpi=300, bbox_inches="tight")
    plt.close()

    series_scores = out.groupby("series_uid", dropna=False)["score"].mean().sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 5))
    plt.barh([str(i)[:22] for i in series_scores.index[::-1]], series_scores.values[::-1])
    plt.xlabel("Score medio")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_series_suspeitas.png", dpi=300, bbox_inches="tight")
    plt.close()

    if len(out):
        row = out.sort_values("score", ascending=False).iloc[0]
        patch, _, _, _ = load_patch_from_meta_row(row, rois_by_sop, args.series_zip_dir, args.patch_size, dataset_type)
        gerar_figura_decomposicao(patch, fig_dir / "figura_poster_exemplo.png")
        gerar_figura_decomposicao(patch, fig_dir / "decomposicao_low_high_gradient_wavelet.png")


def save_gradient_feature_comparison(comparison_df, out_dir, out_dirs):
    rows = []
    if not comparison_df.empty:
        for row in comparison_df.itertuples(index=False):
            versao = getattr(row, "versao")
            usa_wavelet = bool(getattr(row, "usa_wavelet", versao in {"com_wavelet", "wavelet_unet"}))
            usa_gradiente = bool(getattr(row, "usa_gradiente", False))
            usa_unet = bool(getattr(row, "usa_unet", versao == "wavelet_unet"))
            label = "Modelo"
            if usa_wavelet:
                label += " com Wavelet"
            else:
                label += " sem Wavelet"
            if usa_gradiente:
                label += " + Gradiente"
            if usa_unet:
                label += " + U-Net"
            rows.append({
                "modelo": label,
                "versao": versao,
                "usa_wavelet": usa_wavelet,
                "usa_gradiente": usa_gradiente,
                "usa_unet": usa_unet,
                "taxa_suspeita": float(getattr(row, "taxa_pred_nodulo", 0.0)),
                "score_medio": float(getattr(row, "score_medio", 0.0)),
            })
    gradient_df = pd.DataFrame(rows)
    csv_root = Path(out_dir) / "comparacao_features_gradiente.csv"
    csv_structured = out_dirs["csv"] / "comparacao_features_gradiente.csv"
    write_csv_compat(gradient_df, csv_root, csv_structured)

    fig_path = out_dirs["figuras"] / "comparacao_features_gradiente.png"
    if not gradient_df.empty:
        x = np.arange(len(gradient_df))
        width = 0.35
        plt.figure(figsize=(9, 4))
        plt.bar(x - width / 2, gradient_df["taxa_suspeita"], width, label="taxa_suspeita")
        plt.bar(x + width / 2, gradient_df["score_medio"], width, label="score_medio")
        plt.xticks(x, gradient_df["modelo"], rotation=18, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
    return {"csv": str(csv_root), "figura": str(fig_path)}


def write_execution_report(summary, comparison_df, report_path, root_report_path=None):
    lines = [
        "# Relatorio de execucao",
        "",
        f"- Dataset usado: `{summary['fonte']}`",
        f"- Tipo detectado: `{summary['inference_dataset_type']}`",
        f"- Pre-processamento: `{summary['preprocessing']}`",
        f"- Pipeline ativo: `{summary.get('pipeline_ativo', '')}`",
        f"- U-Net: `{summary.get('unet', '')}`",
        f"- Modo de interpretacao: `{summary.get('interpretacao', '')}`",
        f"- Modelo treinado em: `{summary['training_dataset_type']}`",
        f"- Modelo usado: `{summary['versao_usada']}`",
        f"- Threshold usado: `{summary['threshold']:.6f}`",
        f"- Numero de series: `{summary['total_series']}`",
        f"- Numero de imagens: `{summary['total_imagens']}`",
        f"- Figuras: `{summary['figuras_artigo']}`",
        f"- Overlays: `{summary.get('pasta_overlays', '')}`",
        "",
        "## Gradiente em escala de cinza",
        "",
        "- Bloco Sobel habilitado: `sim`",
        "- Numero de features de gradiente: `23`",
        "- O gradiente captura bordas e transicoes locais em escala de cinza, complementando Wavelet, Fourier e filtros passa-alta.",
        "- Impacto observado: comparar `comparacao_features_gradiente.csv` e `comparacao_features_gradiente.png` nos resultados gerados.",
        "",
        "## Taxa suspeita por versao",
        "",
    ]
    if not comparison_df.empty:
        for row in comparison_df.itertuples(index=False):
            lines.append(
                f"- `{row.versao}`: taxa_suspeita={row.taxa_pred_nodulo:.4f}, "
                f"score_medio={row.score_medio:.4f}"
            )
    if summary.get("domain_shift_warning"):
        lines += ["", "## Aviso de domain shift", "", summary["domain_shift_message"]]
    lines += [
        "",
        "## Principais arquivos",
        "",
        f"- `{summary['predicoes_imagens']}`",
        f"- `{summary['predicoes_series']}`",
        f"- `{summary['comparacao_filtros']}`",
        f"- `{summary.get('comparacao_features_gradiente', '')}`",
        f"- `{summary['figuras_artigo']}`",
    ]
    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    if root_report_path is not None:
        root_report_path.write_text(text, encoding="utf-8")


def run_figures_mode(args, out_dir):
    out_dirs = getattr(args, "output_dirs", prepare_output_dirs(out_dir))
    pred_path = out_dirs["csv"] / "predicoes_todas_as_imagens.csv"
    comp_path = out_dirs["csv"] / "comparacao_filtros_wavelet_unet_inferencia.csv"
    if not pred_path.exists():
        pred_path = out_dir / "predicoes_todas_as_imagens.csv"
    if not comp_path.exists():
        comp_path = out_dir / "comparacao_filtros_wavelet_unet_inferencia.csv"
    if not pred_path.exists() or not comp_path.exists():
        raise RuntimeError("CSV de inferencia nao encontrado. Rode --mode infer antes de --mode figures.")
    out = pd.read_csv(pred_path)
    comparison_df = pd.read_csv(comp_path)
    threshold = float(out["threshold"].iloc[0]) if "threshold" in out.columns and len(out) else 0.5
    rois_by_sop = {}
    generate_article_figures(out, comparison_df, out_dirs, threshold, args.dataset_type, rois_by_sop, args)
    save_gradient_feature_comparison(comparison_df, out_dir, out_dirs)
    print(f"Figuras salvas em: {out_dirs['figuras']}")


def run_threshold_mode(args, out_dir):
    pred_path = Path(out_dir) / "predicoes_todas_as_imagens.csv"
    if not pred_path.exists() and hasattr(args, "output_dirs"):
        pred_path = args.output_dirs["csv"] / "predicoes_todas_as_imagens.csv"
    if not pred_path.exists():
        raise RuntimeError("CSV de predicoes nao encontrado para calcular threshold.")
    df = pd.read_csv(pred_path)
    if "y_true" not in df.columns or len(df["y_true"].unique()) < 2:
        raise RuntimeError("Nao ha y_true com duas classes para calcular threshold.")
    y_true = df["y_true"].to_numpy()
    score = df["score"].to_numpy()
    threshold, f1_best = best_f1_threshold(y_true, score)
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    out_dirs = getattr(args, "output_dirs", prepare_output_dirs(out_dir))
    result = {"threshold": threshold, "f1": f1_best}
    (out_dirs["relatorios"] / "threshold_otimo.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (Path(out_dir) / "threshold_otimo.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_dirs["figuras"] / "precision_recall_threshold.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1[:-1] if len(f1) > len(thresholds) else f1[:len(thresholds)])
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(out_dirs["figuras"] / "f1_vs_threshold.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Melhor threshold: {threshold:.6f}")


def save_segmentation_examples(meta, out_dir, patch_size, img_size, series_zip_dir, dataset_type="lidc_ct", n=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = load_unet_model(out_dir, device)
    if unet_model is None or len(meta) == 0:
        return

    examples = meta.head(n)
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    if len(examples) == 1:
        axes = axes[None, :]

    for i, row in enumerate(examples.itertuples(index=False)):
        zip_path = Path(row.zip)
        if not zip_path.is_absolute():
            zip_path = series_zip_dir / row.zip
        ds = read_dicom_from_source(zip_path, row.member)
        img = dicom_to_uint8(ds, dataset_type)
        cx = int(getattr(row, "center_x", img.shape[1] // 2))
        cy = int(getattr(row, "center_y", img.shape[0] // 2))
        patch, _ = crop_centered(img, np.zeros_like(img, dtype=np.uint8), cx, cy, patch_size)

        resized = cv2.resize(patch, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        with torch.no_grad():
            prob = torch.sigmoid(unet_model(torch.from_numpy(resized[None, None]).to(device))).cpu().numpy()[0, 0]
        pred_mask = cv2.resize(prob, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)

        axes[i, 0].imshow(patch, cmap="gray")
        axes[i, 0].set_title(f"Patch {row.label}")
        axes[i, 1].imshow(pred_mask, cmap="magma")
        axes[i, 1].set_title("U-Net sigmoid")
        axes[i, 2].imshow(patch, cmap="gray")
        axes[i, 2].imshow(pred_mask > 0.5, alpha=0.35, cmap="Reds")
        axes[i, 2].set_title("Mascara prevista")
        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "exemplos_segmentacao.png", dpi=150)
    plt.close()


def train_classifier(args, rois_by_sop, out_dir):
    # Pipeline fisico multi-dominio:
    # espacial + fuzzy + low/high-pass + gradiente + Fourier + Wavelet + morfologia U-Net.
    # O meta-modelo foi removido porque a fusao ja e feita no espaco de features
    # e o stacking nao trouxe ganho consistente na validacao por series_uid.
    state_random = random.getstate()
    state_numpy = np.random.get_state()
    try:
        random.seed(args.random_state)
        np.random.seed(args.random_state)
        X_wavelet, y, meta, wavelet_feature_names = load_data(args, rois_by_sop, out_dir, use_wavelet=True, suffix="_wavelet")
        if args.dataset_type == "mammography" and is_rsna_dataset(args.series_zip_dir):
            X_no_wavelet, no_wavelet_feature_names = select_without_wavelet_features(X_wavelet, wavelet_feature_names)
            y2 = y.copy()
            meta_no = meta.copy()
            meta_no.to_csv(out_dir / "amostras_features_sem_wavelet.csv", index=False)
            if hasattr(args, "output_dirs"):
                meta_no.to_csv(args.output_dirs["csv"] / "amostras_features_sem_wavelet.csv", index=False)
        else:
            random.setstate(state_random)
            np.random.set_state(state_numpy)
            random.seed(args.random_state)
            np.random.seed(args.random_state)
            X_no_wavelet, y2, meta_no, no_wavelet_feature_names = load_data(args, rois_by_sop, out_dir, use_wavelet=False, suffix="_sem_wavelet")
    finally:
        random.setstate(state_random)
        np.random.set_state(state_numpy)

    if len(y) != len(y2) or not np.array_equal(y, y2):
        raise RuntimeError("Amostras inconsistentes entre as versoes com e sem Wavelet.")
    if not meta[["zip", "member", "series_uid", "sop_uid", "center_x", "center_y"]].equals(
        meta_no[["zip", "member", "series_uid", "sop_uid", "center_x", "center_y"]]
    ):
        raise RuntimeError("Meta inconsistente entre as versoes com e sem Wavelet.")

    class_counts = Counter(y.tolist())
    if len(class_counts) < 2:
        raise RuntimeError(
            "Nao e possivel treinar nem calcular acertos sem duas classes no ground truth. "
            f"Classes encontradas: {dict(class_counts)}. "
            "Para comparar rede neural com filtros vs sem filtros em cancer de mama, informe uma base "
            "com rotulos benigno/maligno ou um CSV de anotacoes ligado aos pacientes/series/imagens."
        )

    train_idx, test_idx = split_data(
        X_wavelet,
        y,
        meta,
        test_size=args.test_size,
        random_state=args.random_state,
        n_candidates=args.split_candidates,
    )

    split_meta = meta.copy()
    split_meta["split"] = "unused"
    split_meta.loc[train_idx, "split"] = "train"
    split_meta.loc[test_idx, "split"] = "test"
    split_meta.to_csv(out_dir / "split_por_series_uid.csv", index=False)

    X_train_w, X_test_w = X_wavelet[train_idx], X_wavelet[test_idx]
    X_train_wo, X_test_wo = X_no_wavelet[train_idx], X_no_wavelet[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_meta = meta.iloc[test_idx].reset_index(drop=True)

    X_no_filter, no_filter_feature_names = select_no_filter_features(X_wavelet, wavelet_feature_names)
    X_train_nf, X_test_nf = X_no_filter[train_idx], X_no_filter[test_idx]
    print("\n=== Comparacao rede neural: sem filtros vs com filtros ===")
    neural_filter_comparison = train_neural_filter_comparison(
        X_train_nf,
        X_test_nf,
        X_train_w,
        X_test_w,
        y_train,
        y_test,
        out_dir,
        random_state=args.random_state,
    )
    if hasattr(args, "output_dirs"):
        neural_filter_comparison.to_csv(
            args.output_dirs["csv"] / "comparacao_rede_neural_filtros.csv",
            index=False,
        )
        shutil.copy2(
            out_dir / "predicoes_rede_neural_filtros.csv",
            args.output_dirs["csv"] / "predicoes_rede_neural_filtros.csv",
        )
        shutil.copy2(
            out_dir / "relatorio_rede_neural_filtros.txt",
            args.output_dirs["relatorios"] / "relatorio_rede_neural_filtros.txt",
        )
    print(neural_filter_comparison.to_string(index=False))

    print("\n=== Features morfologicas da U-Net ===")
    use_unet = bool(getattr(args, "use_unet", should_use_unet(args.dataset_type)))
    if use_unet:
        X_unet = build_unet_feature_matrix(meta, out_dir, args.patch_size, args.img_size, args.series_zip_dir, args.dataset_type)
        X_unet_train, X_unet_test = X_unet[train_idx], X_unet[test_idx]
    else:
        print("U-Net desativada neste pipeline.")
        X_unet = None
        X_unet_train = X_unet_test = None

    def fit_variant(Xtr, Xte, variant_name, variant_key, report_name, cm_name):
        scaler, best_name, best_model, results = train_best_tabular_model(
            Xtr,
            y_train,
            Xte,
            y_test,
            random_state=args.random_state,
        )
        score = results[best_name]["score"]
        pred = results[best_name]["pred"]
        metrics = evaluate_predictions(
            y_test,
            pred,
            score,
            out_dir,
            report_name,
            cm_name,
            f"{variant_name} ({best_name})",
            threshold=results[best_name]["threshold"],
        )
        return {
            "variant": variant_name,
            "variant_key": variant_key,
            "scaler": scaler,
            "best_name": best_name,
            "best_model": best_model,
            "score": score,
            "pred": pred,
            "threshold": results[best_name]["threshold"],
            "feature_names": (
                no_wavelet_feature_names if variant_key == "sem_wavelet"
                else wavelet_feature_names + unet_prediction_feature_names() if variant_key == "wavelet_unet"
                else wavelet_feature_names
            ),
            "n_features_expected": int(Xtr.shape[1]),
            "metrics": metrics,
        }

    print("\n=== Modelo sem Wavelet ===")
    variant_wo = fit_variant(
        X_train_wo,
        X_test_wo,
        "Sem Wavelet",
        "sem_wavelet",
        "relatorio_modelo_sem_wavelet.txt",
        "matriz_confusao_sem_wavelet.png",
    )

    print("\n=== Modelo com Wavelet ===")
    print("=== Modelo com Wavelet + Gradiente ===")
    variant_w = fit_variant(
        X_train_w,
        X_test_w,
        "Com Wavelet + Gradiente",
        "com_wavelet",
        "relatorio_modelo_com_wavelet.txt",
        "matriz_confusao_com_wavelet.png",
    )

    variant_wu = None
    if use_unet:
        print("\n=== Modelo com Wavelet + Gradiente + U-Net ===")
        X_train_wu = np.hstack([X_train_w, X_unet_train])
        X_test_wu = np.hstack([X_test_w, X_unet_test])
        variant_wu = fit_variant(
            X_train_wu,
            X_test_wu,
            "Wavelet + Gradiente + U-Net",
            "wavelet_unet",
            "relatorio_modelo_wavelet_unet.txt",
            "matriz_confusao_wavelet_unet.png",
        )

    # Seleciona o melhor modelo final sem stacking: uma unica familia de features
    # ja carrega suficiente informacao espacial, espectral e morfologica.
    variants = [variant_wo, variant_w] + ([variant_wu] if variant_wu is not None else [])
    best_variant = max(variants, key=lambda v: (v["metrics"]["pr_auc"], v["metrics"]["roc_auc"]))
    variant_name_to_key = {
        "Sem Wavelet": "sem_wavelet",
        "Com Wavelet": "com_wavelet",
        "Com Wavelet + Gradiente": "com_wavelet",
        "Wavelet + U-Net": "wavelet_unet",
        "Wavelet + Gradiente + U-Net": "wavelet_unet",
    }
    best_variant_key = variant_name_to_key[best_variant["variant"]]
    print(f"\n=== Melhor versao final: {best_variant['variant']} ({best_variant['best_name']}) ===")

    plot_metricas(y_test, best_variant["score"], out_dir / "curva_roc_pr.png")
    plot_comparative_curves(
        {
            "sem_wavelet": variant_wo["score"],
            "com_wavelet": variant_w["score"],
            **({"wavelet_unet": variant_wu["score"]} if variant_wu is not None else {}),
        },
        y_test,
        out_dir,
    )

    sample_row = test_meta[test_meta["label"] == "NODULO"].iloc[0] if (test_meta["label"] == "NODULO").any() else test_meta.iloc[0]
    patch, mask_gt, _, _ = load_patch_from_meta_row(sample_row, rois_by_sop, args.series_zip_dir, args.patch_size, args.dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = load_unet_model(out_dir, device)
    _, mask_pred = predict_unet_maps(patch, unet_model, device, args.img_size)

    gerar_figuras_artigo(
        patch,
        mask_gt=mask_gt,
        mask_pred=mask_pred,
        save_path=out_dir / "figura_principal.png",
    )
    gerar_figura_poster(
        patch,
        save_path=out_dir / "poster_estilo.png",
    )
    plot_histograma_intensidades(
        patch,
        mask_roi=mask_gt,
        save_path=out_dir / "histograma_features.png",
    )

    imagens = []
    masks_gt = []
    masks_pred = []
    for row in test_meta.head(min(8, len(test_meta))).itertuples(index=False):
        patch_i, mask_gt_i, _, _ = load_patch_from_meta_row(row, rois_by_sop, args.series_zip_dir, args.patch_size, args.dataset_type)
        _, mask_pred_i = predict_unet_maps(patch_i, unet_model, device, args.img_size)
        imagens.append(patch_i)
        masks_gt.append(mask_gt_i)
        masks_pred.append(mask_pred_i)
    gerar_exemplos_segmentacao(
        imagens,
        masks_gt=masks_gt,
        masks_pred=masks_pred,
        save_path=out_dir / "exemplos_segmentacao.png",
        nmax=8,
    )

    pd.DataFrame({
        "y_true": y_test,
        "score_sem_wavelet": variant_wo["score"],
        "score_com_wavelet": variant_w["score"],
        **({"score_wavelet_unet": variant_wu["score"]} if variant_wu is not None else {}),
        "pred_sem_wavelet": variant_wo["pred"],
        "pred_com_wavelet": variant_w["pred"],
        **({"pred_wavelet_unet": variant_wu["pred"]} if variant_wu is not None else {}),
    }).to_csv(out_dir / "comparacao_wavelet.csv", index=False)

    serie_eval = test_meta.copy()
    serie_eval["score_sem_wavelet"] = variant_wo["score"]
    serie_eval["score_com_wavelet"] = variant_w["score"]
    serie_eval["pred_sem_wavelet"] = variant_wo["pred"]
    serie_eval["pred_com_wavelet"] = variant_w["pred"]
    agg_spec = {
        "n_amostras": ("series_uid", "size"),
        "n_nodulo": ("label", lambda s: int((s == "NODULO").sum())),
        "n_sem_nodulo": ("label", lambda s: int((s == "SEM_NODULO").sum())),
        "score_sem_wavelet": ("score_sem_wavelet", "mean"),
        "score_com_wavelet": ("score_com_wavelet", "mean"),
    }
    if variant_wu is not None:
        serie_eval["score_wavelet_unet"] = variant_wu["score"]
        serie_eval["pred_wavelet_unet"] = variant_wu["pred"]
        agg_spec["score_wavelet_unet"] = ("score_wavelet_unet", "mean")
    serie_eval = serie_eval.groupby("series_uid", dropna=False).agg(
        **agg_spec
    ).reset_index()
    serie_eval.to_csv(out_dir / "avaliacao_por_serie.csv", index=False)

    tabular_pack = {
        "sem_wavelet": {
            "scaler": variant_wo["scaler"],
            "model": variant_wo["best_model"],
            "name": variant_wo["best_name"],
            "threshold": variant_wo["threshold"],
            "feature_names": variant_wo["feature_names"],
            "n_features_expected": variant_wo["n_features_expected"],
        },
        "com_wavelet": {
            "scaler": variant_w["scaler"],
            "model": variant_w["best_model"],
            "name": variant_w["best_name"],
            "threshold": variant_w["threshold"],
            "feature_names": variant_w["feature_names"],
            "n_features_expected": variant_w["n_features_expected"],
        },
    }
    if variant_wu is not None:
        tabular_pack["wavelet_unet"] = {
            "scaler": variant_wu["scaler"],
            "model": variant_wu["best_model"],
            "name": variant_wu["best_name"],
            "threshold": variant_wu["threshold"],
            "feature_names": variant_wu["feature_names"],
            "n_features_expected": variant_wu["n_features_expected"],
        }

    model_pack = {
        "best_variant": best_variant_key,
        "tabular": tabular_pack,
        "class_names": CLASS_NAMES,
        "training_dataset_type": args.dataset_type,
        "dataset_type": args.dataset_type,
        "pipeline_version": "v2_gradient_wavelet",
        "model": best_variant["best_model"],
        "scaler": best_variant["scaler"],
        "feature_names": best_variant["feature_names"],
        "feature_blocks": [
            "spatial",
            "fuzzy",
            "low_high",
            "gradient",
            "fft",
            "wavelet",
            "unet_morph",
        ],
        "gradient_feature_names": FEATURE_REGISTRY["gradient"],
    }
    joblib.dump(model_pack, out_dir / "modelos_integrados_sem_randomforest.joblib")
    if hasattr(args, "output_dirs"):
        joblib.dump(model_pack, args.output_dirs["modelos"] / "modelos_integrados_sem_randomforest.joblib")

    np.save(out_dir / "X_features_wavelet.npy", X_wavelet)
    np.save(out_dir / "X_features_sem_wavelet.npy", X_no_wavelet)
    if X_unet is not None:
        np.save(out_dir / "X_unet_features.npy", X_unet)
    np.save(out_dir / "y_features.npy", y)

    metrics_payload = {
        "sem_wavelet": {k: v for k, v in variant_wo["metrics"].items() if k != "score"},
        "com_wavelet": {k: v for k, v in variant_w["metrics"].items() if k != "score"},
        "comparacao_rede_neural_filtros": neural_filter_comparison.to_dict(orient="records"),
        "features_sem_filtros": no_filter_feature_names,
        "features_com_filtros": wavelet_feature_names,
        "best_variant": best_variant["variant"],
        "best_model": best_variant["best_name"],
        "best_threshold": best_variant["threshold"],
    }
    if variant_wu is not None:
        metrics_payload["wavelet_unet"] = {k: v for k, v in variant_wu["metrics"].items() if k != "score"}
    (out_dir / "metricas_classificador.json").write_text(json.dumps(metrics_payload, indent=2))

    trained_tabular = {
        "sem_wavelet": {
            "scaler": variant_wo["scaler"],
            "model": variant_wo["best_model"],
            "name": variant_wo["best_name"],
            "threshold": variant_wo["threshold"],
            "feature_names": variant_wo["feature_names"],
            "n_features_expected": variant_wo["n_features_expected"],
        },
        "com_wavelet": {
            "scaler": variant_w["scaler"],
            "model": variant_w["best_model"],
            "name": variant_w["best_name"],
            "threshold": variant_w["threshold"],
            "feature_names": variant_w["feature_names"],
            "n_features_expected": variant_w["n_features_expected"],
        },
    }
    if variant_wu is not None:
        trained_tabular["wavelet_unet"] = {
            "scaler": variant_wu["scaler"],
            "model": variant_wu["best_model"],
            "name": variant_wu["best_name"],
            "threshold": variant_wu["threshold"],
            "feature_names": variant_wu["feature_names"],
            "n_features_expected": variant_wu["n_features_expected"],
        }

    return {
        "best_variant": best_variant["variant"],
        "best_name": best_variant["best_name"],
        "tabular": trained_tabular,
        "sem_wavelet": variant_wo,
        "com_wavelet": variant_w,
        "wavelet_unet": variant_wu,
        "best_variant_key": best_variant_key,
        "training_dataset_type": args.dataset_type,
    }


def infer_all_images(args, rois_by_sop, out_dir, trained=None):
    """
    Executa inferencia em todas as fatias disponiveis, uma por uma.
    Gera predito por slice e resumo por serie.
    """
    model_dir = getattr(args, "model_dir", None) or out_dir
    training_dataset_type = "unknown"
    if trained is None:
        model_pack, model_path = resolve_model_pack(model_dir)
        training_dataset_type = model_pack.get("training_dataset_type", model_pack.get("dataset_type", "unknown"))
        if not _is_new_pipeline_pack(model_pack):
            print("Modelo antigo detectado. Recomenda-se retreinar para remover incompatibilidades.")
        if "tabular" in model_pack:
            trained = {
                "best_variant": model_pack["best_variant"],
                "tabular": model_pack["tabular"],
                "training_dataset_type": training_dataset_type,
            }
        else:
            trained = {
                "best_variant": model_pack["best_variant"],
                "tabular": {
                    "sem_wavelet": model_pack["sem_wavelet"],
                    "com_wavelet": model_pack["com_wavelet"],
                    "wavelet_unet": model_pack["wavelet_unet"],
                },
                "training_dataset_type": training_dataset_type,
            }
        if model_pack.get("model") is not None and model_pack.get("scaler") is not None:
            trained["active"] = {
                "model": model_pack["model"],
                "scaler": model_pack["scaler"],
                "feature_names": model_pack.get("feature_names", []),
                "dataset_type": model_pack.get("dataset_type", training_dataset_type),
                "pipeline_version": model_pack.get("pipeline_version", "legacy"),
            }
    else:
        training_dataset_type = trained.get("training_dataset_type", getattr(args, "dataset_type", "unknown"))
    if training_dataset_type == "unknown" and "LIDC" in str(model_dir).upper():
        training_dataset_type = "lidc_ct"

    best_variant = trained["best_variant"]
    variant_name_to_key = {
        "Sem Wavelet": "sem_wavelet",
        "Com Wavelet": "com_wavelet",
        "Com Wavelet + Gradiente": "com_wavelet",
        "Wavelet + U-Net": "wavelet_unet",
        "Wavelet + Gradiente + U-Net": "wavelet_unet",
    }
    best_variant = variant_name_to_key.get(best_variant, best_variant)
    use_unet = bool(getattr(args, "use_unet", should_use_unet(args.dataset_type)))
    if not use_unet:
        print("U-Net desativada neste pipeline.")
        if best_variant == "wavelet_unet":
            if "com_wavelet" in trained["tabular"]:
                print("Modelo wavelet_unet ignorado neste pipeline; usando com_wavelet.")
                best_variant = "com_wavelet"
            elif "sem_wavelet" in trained["tabular"]:
                print("Modelo wavelet_unet ignorado neste pipeline; usando sem_wavelet.")
                best_variant = "sem_wavelet"
    variant_pack = trained["tabular"][best_variant]

    X_no_wavelet, y_full, meta_full, full_no_wavelet_names = build_full_image_dataset(
        args.series_zip_dir,
        rois_by_sop,
        args.patch_size,
        max_series=args.max_series,
        use_wavelet=False,
        dataset_type=args.dataset_type,
    )
    X_wavelet, _, _, full_wavelet_names = build_full_image_dataset(
        args.series_zip_dir,
        rois_by_sop,
        args.patch_size,
        max_series=args.max_series,
        use_wavelet=True,
        dataset_type=args.dataset_type,
    )
    X_unet_full = None
    if use_unet and "wavelet_unet" in trained["tabular"]:
        X_unet_full = build_unet_feature_matrix(meta_full, model_dir, args.patch_size, args.img_size, args.series_zip_dir, args.dataset_type)

    variant_outputs = {}
    comparison_rows = []
    has_ground_truth = len(np.unique(y_full)) >= 2
    for variant_key, pack in trained["tabular"].items():
        if variant_key == "wavelet_unet" and not use_unet:
            continue
        if variant_key == "wavelet_unet" and X_unet_full is None:
            continue
        X_variant = make_variant_feature_matrix(
            variant_key,
            X_no_wavelet,
            X_wavelet,
            X_unet_full,
            pack,
            dataset_type=args.dataset_type,
        )
        X_scaled = pack["scaler"].transform(X_variant)
        variant_score = model_scores(pack["model"], X_scaled)
        if str(args.threshold).lower() == "auto":
            variant_threshold = best_f1_threshold(y_full, variant_score)[0] if has_ground_truth else 0.5
        elif args.threshold is not None:
            variant_threshold = float(args.threshold)
        else:
            variant_threshold = float(pack.get("threshold", 0.5))
        variant_pred = (variant_score >= variant_threshold).astype(int)
        variant_outputs[variant_key] = {
            "score": variant_score,
            "pred": variant_pred,
            "threshold": variant_threshold,
            "summary": summarize_binary_predictions(y_full, variant_score, variant_pred),
        }
        expected_names = pack.get("feature_names")
        expected_count = pack.get("n_features_expected") or getattr(pack["scaler"], "n_features_in_", None)
        if not expected_names and expected_count is not None:
            expected_names = legacy_feature_names_for_variant(variant_key, int(expected_count))
        expected_names = expected_names or feature_names_for_variant(variant_key)
        comparison_rows.append({
            "versao": variant_key,
            "modelo": pack.get("name", ""),
            "threshold": variant_threshold,
            "usa_wavelet": bool(variant_key in {"com_wavelet", "wavelet_unet"}),
            "usa_gradiente": bool(any(str(name).startswith("gradient_") for name in expected_names)),
            "usa_unet": bool(variant_key == "wavelet_unet"),
            **variant_outputs[variant_key]["summary"],
        })

    score = variant_outputs[best_variant]["score"]
    threshold = variant_outputs[best_variant]["threshold"]
    pred = variant_outputs[best_variant]["pred"]
    domain_shift_warning = (
        training_dataset_type not in {"unknown", args.dataset_type}
        and args.dataset_type != "generic_dicom"
    )
    interpretacao_modo = "anomaly_detection" if domain_shift_warning or not has_ground_truth else "supervised_inference"

    out = meta_full.copy()
    out["y_true"] = y_full
    for variant_key, values in variant_outputs.items():
        out[f"score_{variant_key}"] = values["score"]
        out[f"pred_{variant_key}"] = values["pred"]
        out[f"threshold_{variant_key}"] = values["threshold"]
    out["score"] = score
    out["threshold"] = threshold
    out["pred"] = pred
    out["score_suspeita"] = score
    out["regiao_suspeita"] = pred
    out["interpretacao"] = np.where(pred == 1, "suspeito", "baixo_score")
    out["dataset_type"] = args.dataset_type
    out["preprocessing"] = preprocessing_name(args.dataset_type)
    out["domain_shift_warning"] = domain_shift_warning
    out["domain_shift"] = domain_shift_warning
    out["modo"] = interpretacao_modo
    out_dirs = getattr(args, "output_dirs", prepare_output_dirs(out_dir))
    write_csv_compat(
        out,
        out_dir / "predicoes_todas_as_imagens.csv",
        out_dirs["csv"] / "predicoes_todas_as_imagens.csv",
    )

    comparison_df = pd.DataFrame(comparison_rows)
    write_csv_compat(
        comparison_df,
        out_dir / "comparacao_filtros_wavelet_unet_inferencia.csv",
        out_dirs["csv"] / "comparacao_filtros_wavelet_unet_inferencia.csv",
    )
    gradient_comparison_paths = save_gradient_feature_comparison(comparison_df, out_dir, out_dirs)

    serie_eval = out.groupby("series_uid", dropna=False).agg(
        n_amostras=("series_uid", "size"),
        n_nodulo=("y_true", "sum"),
        score_medio=("score", "mean"),
        taxa_pred_nodulo=("pred", "mean"),
    ).reset_index()
    write_csv_compat(
        serie_eval,
        out_dir / "predicoes_todas_as_series.csv",
        out_dirs["csv"] / "predicoes_todas_as_series.csv",
    )

    print("\n=== Inferencia em todas as imagens ===")
    print(f"Versao usada: {best_variant}")
    print(f"Threshold usado: {threshold:.6f}")
    print("\n=== Comparacao filtros / Wavelet / U-Net ===")
    print(comparison_df.to_string(index=False))
    saved_overlays = []
    # Salvar uma PNG por série por padrão, ou por fatia se solicitado explicitamente.
    if getattr(args, "save_overlays", "none") != "none":
        target_root = Path(args.save_target_dir) if getattr(args, "save_target_dir", None) else out_dirs["overlays"]
        target_root.mkdir(parents=True, exist_ok=True)

        if args.save_overlays in {"top", "series", "all"}:
            grouped = list(out.groupby("series_uid", dropna=False))
            if args.save_overlays == "top":
                ranked = out.groupby("series_uid", dropna=False)["score"].mean().sort_values(ascending=False).head(10).index
                grouped = [(uid, group) for uid, group in grouped if uid in set(ranked)]
            for series_uid, group in grouped:
                saved = save_score_overlay_series(group, target_root, threshold, series_uid)
                saved_overlays.append(str(saved))
                print(f"Overlay salvo: {saved}")
        elif args.save_overlays == "slice":
            for series_uid, group in out.groupby("series_uid"):
                series_dir = target_root / str(series_uid)
                series_dir.mkdir(parents=True, exist_ok=True)
                for idx, (_, r) in enumerate(group.iterrows()):
                    try:
                        patch, patch_mask, img_full, gt_mask_full = load_patch_from_meta_row(r, rois_by_sop, args.series_zip_dir, args.patch_size, args.dataset_type)
                    except Exception:
                        continue
                    pred_mask = np.zeros_like(patch, dtype=np.uint8)
                    unet_path = Path(model_dir) / "unet_lidc_best.pt"
                    if unet_path.exists():
                        try:
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            unet_model = UNetSmall().to(device)
                            unet_model.load_state_dict(torch.load(unet_path, map_location=device))
                            unet_model.eval()
                            p_resized = cv2.resize(patch, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
                            t = torch.from_numpy((p_resized.astype(np.float32) / 255.0)[None, None]).to(device)
                            with torch.no_grad():
                                logits = unet_model(t)
                                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
                            probs_small = cv2.resize(probs, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
                            pred_mask = (probs_small > 0.5).astype(np.uint8)
                        except Exception:
                            pred_mask = np.zeros_like(patch, dtype=np.uint8)
                    member_name = Path(str(r.member)).stem
                    out_file = series_dir / f"overlay_{series_uid}_{idx:04d}_{member_name}.png"
                    save_overlay_image(patch, patch_mask, pred_mask, out_file)
                    saved_overlays.append(str(out_file))
                    if idx == 0:
                        print(f"Primeiro overlay da série salvo: {out_file}")

    generate_article_figures(out, comparison_df, out_dirs, threshold, args.dataset_type, rois_by_sop, args)
    if domain_shift_warning and training_dataset_type == "lidc_ct" and args.dataset_type == "mammography":
        cross_domain_message = "Modelo treinado em CT pulmonar aplicado em mamografia. Interpretar como anomaly detection, não diagnóstico."
    elif domain_shift_warning:
        cross_domain_message = (
            f"Modelo treinado em {training_dataset_type} aplicado em {args.dataset_type}. "
            "Interpretar como anomaly detection, não diagnóstico."
        )
    else:
        cross_domain_message = ""

    summary = {
        "fonte": str(args.series_zip_dir),
        "saida": str(out_dir),
        "modelo": str(model_dir),
        "versao_usada": best_variant,
        "training_dataset_type": training_dataset_type,
        "inference_dataset_type": args.dataset_type,
        "preprocessing": preprocessing_name(args.dataset_type),
        "pipeline_ativo": pipeline_name(args.dataset_type),
        "unet": "ativado" if use_unet else "desativado",
        "domain_shift_warning": bool(domain_shift_warning),
        "domain_shift": bool(domain_shift_warning),
        "interpretacao": interpretacao_modo,
        "message": cross_domain_message,
        "domain_shift_message": cross_domain_message,
        "threshold": threshold,
        "max_series": args.max_series,
        "total_imagens": int(len(out)),
        "total_series": int(out["series_uid"].nunique(dropna=False)),
        "predicoes_imagens": str(out_dir / "predicoes_todas_as_imagens.csv"),
        "predicoes_series": str(out_dir / "predicoes_todas_as_series.csv"),
        "comparacao_filtros": str(out_dir / "comparacao_filtros_wavelet_unet_inferencia.csv"),
        "comparacao_features_gradiente": gradient_comparison_paths["csv"],
        "figura_comparacao_features_gradiente": gradient_comparison_paths["figura"],
        "figuras_artigo": str(out_dirs["figuras"]),
        "metricas_por_versao": {
            key: values["summary"] for key, values in variant_outputs.items()
        },
        "overlays_salvos": int(len(saved_overlays)),
        "pasta_overlays": str((Path(args.save_target_dir) if getattr(args, "save_target_dir", None) else out_dirs["overlays"]))
        if getattr(args, "save_overlays", "none") != "none" else "",
    }
    (out_dir / "resumo_inferencia.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dirs["relatorios"] / "resumo_inferencia.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_execution_report(summary, comparison_df, out_dirs["relatorios"] / "relatorio_execucao.md", out_dir / "relatorio_execucao.md")
    print("\n=== Resultados salvos ===")
    print(f"Dataset detectado: {args.dataset_type}")
    print(f"Pré-processamento: {preprocessing_name(args.dataset_type)}")
    print(f"Pipeline ativo: {pipeline_name(args.dataset_type)}")
    print(f"U-Net: {'ativado' if use_unet else 'desativado'}")
    print(f"Modo: {interpretacao_modo}")
    print(f"Modelo treinado em: {training_dataset_type}")
    if domain_shift_warning:
        print("Aviso: cross-domain inference")
        print(cross_domain_message)
    print(f"CSV imagens: {summary['predicoes_imagens']}")
    print(f"CSV series: {summary['predicoes_series']}")
    print(f"Comparacao filtros: {summary['comparacao_filtros']}")
    print(f"Comparacao features gradiente: {summary['comparacao_features_gradiente']}")
    print(f"Resumo JSON: {out_dir / 'resumo_inferencia.json'}")
    print(f"Figuras salvas em: {summary['figuras_artigo']}")
    if summary["pasta_overlays"]:
        print(f"Overlays: {summary['pasta_overlays']} ({summary['overlays_salvos']} arquivos)")
    print(f"Relatório salvo em: {out_dir / 'relatorio_execucao.md'}")

    if len(np.unique(y_full)) < 2:
        print("Sem anotacoes com duas classes nesta base; metricas supervisionadas foram puladas.")
        return out

    print(classification_report(y_full, pred, target_names=CLASS_NAMES))
    print(f"ROC AUC: {roc_auc_score(y_full, score):.4f}")
    print(f"Precision-Recall AUC: {average_precision_score(y_full, score):.4f}")

    evaluate_predictions(
        y_full,
        pred,
        score,
        out_dir,
        "relatorio_inferencia_todas_as_imagens.txt",
        "matriz_confusao_inferencia_todas_as_imagens.png",
        "Inferencia em todas as imagens",
    )

    plot_metricas(y_full, score, out_dir / "curva_roc_pr_todas_as_imagens.png")
    return out


def main():
    parser = argparse.ArgumentParser(description="Sistema integrado LIDC: U-Net + fuzzy/Fourier + classificador.")
    parser.add_argument(
        "--mode",
        choices=["all", "full", "features", "unet", "infer", "train", "figures", "threshold"],
        default="all",
        help="all/full = treina e depois infere; features = treina classificador; unet = treina U-Net; infer = somente inferencia.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_DATA_ROOT / "extracted")
    parser.add_argument("--dicom-root", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_DATA_ROOT / "extracted_manifest.csv")
    parser.add_argument("--max-series", type=int, default=160)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--neg-prob", type=float, default=0.02)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-candidates", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--dataset-type", choices=sorted(DATASET_TYPES), default="auto")
    parser.add_argument("--force-retrain", action="store_true", help="Ignora modelos antigos e retreina o pipeline atual.")
    parser.add_argument("--threshold", default="0.5",
                        help="Threshold numerico ou 'auto' quando houver y_true.")
    parser.add_argument("--save-overlays", choices=["none", "top", "series", "all", "slice"], default="none",
                        help="Salvar overlays por score: top, series ou all. 'slice' mantém compatibilidade antiga.")
    parser.add_argument("--save-target-dir", type=Path, default=None,
                        help="Diretório alvo para salvar overlays. Se omitido, salva dentro da pasta da série (ou em out_dir se não for possível).")
    args = parser.parse_args()
    if args.mode == "full":
        args.mode = "all"
    if args.mode == "train":
        args.mode = "features"

    if args.dicom_root is not None:
        args.series_zip_dir = args.dicom_root
    elif args.extracted_root.exists():
        args.series_zip_dir = args.extracted_root
    elif (args.data_root / "series_zip").exists():
        args.series_zip_dir = args.data_root / "series_zip"
    else:
        args.series_zip_dir = args.data_root
    xml_zip = args.data_root / "annotations_metadata" / "LIDC-XML-only.zip"
    out_dir = args.out_dir if args.out_dir is not None else args.data_root / "sistema_integrado"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_dirs = prepare_output_dirs(out_dir)
    args.dataset_type = detect_dataset_type(args.series_zip_dir, args.dataset_type)

    print("Data root:", args.data_root)
    rsna_train_csv, rsna_images_dir = resolve_rsna_paths(args.series_zip_dir)
    source_zip_count = len(list(args.series_zip_dir.glob("*.zip")))
    source_dir_count = len([p for p in args.series_zip_dir.iterdir() if p.is_dir()]) if args.series_zip_dir.exists() else 0
    source_dicom_series_count = 0 if rsna_train_csv is not None else count_dicom_series_dirs(args.series_zip_dir)
    print("Fonte:", args.series_zip_dir)
    print("Dataset detectado:", args.dataset_type)
    print("Pré-processamento:", preprocessing_name(args.dataset_type))
    print("Pipeline ativo:", pipeline_name(args.dataset_type))
    print("Series DICOM:", source_dicom_series_count)
    print("ZIPs/pastas:", source_zip_count if source_zip_count else source_dir_count)
    if rsna_train_csv is not None:
        print("RSNA train.csv:", rsna_train_csv)
        print("RSNA imagens:", rsna_images_dir)
    if args.manifest_csv.exists():
        print("Manifest:", args.manifest_csv)
    print("Saida:", out_dir)
    if args.dataset_type == "lidc_ct":
        rois_by_sop = parse_lidc_rois(xml_zip, out_dir / "rois_by_sop.json")
    else:
        # Anotações LIDC são específicas de CT pulmonar. Em mamografia elas não
        # representam ground truth compatível e não devem acionar treino de U-Net.
        rois_by_sop = {}
    print("Fatias com ROI:", len(rois_by_sop))
    if not should_use_unet(args.dataset_type):
        print("U-Net desativada: mamografia não possui máscaras compatíveis")

    if args.mode == "figures":
        run_figures_mode(args, out_dir)
        return
    if args.mode == "threshold":
        run_threshold_mode(args, out_dir)
        return

    # Se nao houver anotações suficientes para treinar (ex.: base ACRIN sem LIDC),
    # evitamos tentar treinar U-Net/classificador automaticamente.
    if args.mode == "unet" and not should_use_unet(args.dataset_type):
        print("U-Net desativada: mamografia não possui máscaras compatíveis")
        print("Não é conceitualmente correto treinar U-Net sem máscaras compatíveis neste domínio.")
        return

    if args.mode in ("all", "unet") and len(rois_by_sop) < 10 and not args.force_retrain:
        model_dir_candidate = args.model_dir if getattr(args, "model_dir", None) else out_dir
        unet_path = Path(model_dir_candidate) / "unet_lidc_best.pt"
        tabular_path = Path(model_dir_candidate) / "modelos_integrados_sem_randomforest.joblib"
        if not unet_path.exists():
            unet_path = Path(model_dir_candidate) / "modelos" / "unet_lidc_best.pt"
        if not tabular_path.exists():
            tabular_path = Path(model_dir_candidate) / "modelos" / "modelos_integrados_sem_randomforest.joblib"
        print(f"Aviso: poucas anotações LIDC detectadas ({len(rois_by_sop)}).")
        if tabular_path.exists() and (unet_path.exists() or not should_use_unet(args.dataset_type)):
            print(f"Modelos pré-treinados encontrados em {model_dir_candidate}; alternando para modo 'infer'.")
            args.mode = "infer"
        else:
            print("Não é possível treinar sem anotações suficientes.\n" \
                  "Opções:\n" \
                  "  1) Rode com --mode infer e forneça --model-dir apontando para modelos treinados.\n" \
                  "  2) Execute em uma base com anotações LIDC ou gere anotações antes de treinar.")
            return

    args.use_unet = should_use_unet(args.dataset_type)
    if args.mode in ("all", "unet") and args.use_unet:
        try:
            best, use_unet = train_unet(args, rois_by_sop, out_dir)
            args.use_unet = use_unet
            print("Melhor Dice U-Net:", best)
        except RuntimeError as exc:
            msg = str(exc)
            if not args.force_retrain and ("Poucas fatias positivas" in msg or "Poucas anotações" in msg):
                model_dir_candidate = args.model_dir if getattr(args, "model_dir", None) else out_dir
                unet_path = Path(model_dir_candidate) / "unet_lidc_best.pt"
                tabular_path = Path(model_dir_candidate) / "modelos_integrados_sem_randomforest.joblib"
                if unet_path.exists() and tabular_path.exists():
                    print("Falha ao treinar U-Net; alternando para inferência com modelos existentes.")
                    args.mode = "infer"
                else:
                    raise
            else:
                raise
    if not getattr(args, "use_unet", False):
        print("U-Net desativada: desempenho insuficiente (Dice muito baixo)")
    if args.mode in ("all", "features"):
        trained = train_classifier(args, rois_by_sop, out_dir)
        if args.mode == "all":
            infer_all_images(args, rois_by_sop, out_dir, trained=trained)
    if args.mode == "infer":
        infer_all_images(args, rois_by_sop, out_dir)


if __name__ == "__main__":
    main()
