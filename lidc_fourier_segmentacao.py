# %% [markdown]
# LIDC-IDRI: segmentacao + Fourier para testar padrao de nodulo
#
# Ideia:
# 1. Ler DICOM dos ZIPs baixados do TCIA.
# 2. Usar os contornos XML oficiais do LIDC para criar mascara da lesao.
# 3. Extrair patches segmentados.
# 4. Medir textura no dominio da frequencia com FFT radial.
# 5. Treinar um classificador simples para NODULO vs SEM_NODULO.
#
# Observacao: CT nao tem "cor" real. O que analisamos e tom/densidade HU,
# bordas, textura e frequencias da imagem.

# %%
# Se faltar dependencia no kernel:
# !pip install pydicom pandas scikit-learn tqdm matplotlib pillow opencv-python

import io
import math
import random
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib.pyplot as plt
import shutil
import datetime


# %%
LIDC_ROOT = Path("/media/angualberto/C8C814BEC814AD26/TCIA/LIDC-IDRI")
SERIES_ZIP_DIR = LIDC_ROOT / "series_zip"
ANNOTATIONS_ZIP = LIDC_ROOT / "annotations_metadata" / "LIDC-XML-only.zip"
OUT_DIR = LIDC_ROOT / "fourier_segmentacao"
OUT_DIR.mkdir(exist_ok=True)

# Use pequeno enquanto o download ainda esta rodando. Depois use None.
MAX_SERIES = 120

# Tamanho do patch ao redor da lesao ou regiao negativa.
PATCH_SIZE = 96

CLASS_NAMES = ["SEM_NODULO", "NODULO"]

print("ZIPs disponiveis:", len(list(SERIES_ZIP_DIR.glob("*.zip"))))
print("Saida:", OUT_DIR)


# %%
def strip_ns(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def direct_child_text(node, name):
    for child in node:
        if strip_ns(child.tag) == name:
            return child.text
    return None


def parse_lidc_rois(xml_zip_path):
    """
    Retorna:
      rois_by_sop[SOPInstanceUID] = lista de dicts:
        {
          "series_uid": ...,
          "points": [(x, y), ...],
          "malignancy": int|None
        }

    O LIDC tem varios radiologistas. Uma mesma fatia pode ter mais de um ROI.
    Aqui mantemos todos; depois juntamos as mascaras da fatia.
    """
    rois_by_sop = defaultdict(list)

    with zipfile.ZipFile(xml_zip_path) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]

        for name in tqdm(xml_names, desc="Lendo XML"):
            try:
                root = ET.fromstring(zf.read(name))
            except ET.ParseError:
                continue

            series_uid = None
            for node in root.iter():
                if strip_ns(node.tag) == "SeriesInstanceUid":
                    series_uid = node.text
                    break
            if not series_uid:
                continue

            for nodule in root.iter():
                if strip_ns(nodule.tag) != "unblindedReadNodule":
                    continue

                malignancy = None
                for child in nodule:
                    if strip_ns(child.tag) == "characteristics":
                        value = direct_child_text(child, "malignancy")
                        malignancy = int(value) if value and value.isdigit() else None

                for roi in nodule:
                    if strip_ns(roi.tag) != "roi":
                        continue
                    inclusion = direct_child_text(roi, "inclusion")
                    sop_uid = direct_child_text(roi, "imageSOP_UID")
                    if not sop_uid or str(inclusion).upper() != "TRUE":
                        continue

                    points = []
                    for edge in roi:
                        if strip_ns(edge.tag) != "edgeMap":
                            continue
                        x = direct_child_text(edge, "xCoord")
                        y = direct_child_text(edge, "yCoord")
                        if x is not None and y is not None:
                            points.append((int(float(x)), int(float(y))))

                    if len(points) >= 3:
                        rois_by_sop[sop_uid].append({
                            "series_uid": series_uid,
                            "points": points,
                            "malignancy": malignancy,
                        })

    return dict(rois_by_sop)


rois_by_sop = parse_lidc_rois(ANNOTATIONS_ZIP)
print("Fatias com ROI:", len(rois_by_sop))


# %%
def dicom_to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def hu_to_uint8_lung(hu, window_min=-1000, window_max=400):
    clipped = np.clip(hu, window_min, window_max)
    return ((clipped - window_min) / (window_max - window_min) * 255.0).astype(np.uint8)


def mask_from_rois(shape, rois):
    mask = np.zeros(shape, dtype=np.uint8)
    for roi in rois:
        pts = np.array(roi["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def crop_centered(img, mask, center_x, center_y, size=96):
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


def random_lung_like_negative(hu, positive_mask, size=96, tries=80):
    """
    Amostra uma regiao negativa dentro de area plausivel de pulmao.
    Usa threshold HU simples para evitar fundo preto/mesa.
    """
    lungish = (hu > -1000) & (hu < 200)
    h, w = hu.shape
    half = size // 2

    ys, xs = np.where(lungish & (positive_mask == 0))
    if len(xs) == 0:
        return None

    for _ in range(tries):
        idx = random.randrange(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        if cx < half or cy < half or cx >= w - half or cy >= h - half:
            continue
        patch_mask = positive_mask[cy-half:cy+half, cx-half:cx+half]
        if patch_mask.sum() == 0:
            img8 = hu_to_uint8_lung(hu)
            empty_mask = np.zeros_like(img8, dtype=np.uint8)
            return crop_centered(img8, empty_mask, cx, cy, size=size)
    return None


# %%
def fft_radial_features(patch, mask=None, bins=24):
    """
    Extrai energia por bandas radiais no dominio Fourier.
    Features:
      - bins de energia radial normalizada
      - razao baixa/media/alta frequencia
    """
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

    feats = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        band = mag[(rr >= lo) & (rr < hi)]
        feats.append(float(band.mean()) if band.size else 0.0)

    feats = np.array(feats, dtype=np.float32)
    feats = feats / (feats.sum() + 1e-6)

    low = feats[: bins // 3].sum()
    mid = feats[bins // 3 : 2 * bins // 3].sum()
    high = feats[2 * bins // 3 :].sum()
    extra = np.array([
        low,
        mid,
        high,
        high / (low + 1e-6),
        mid / (low + 1e-6),
    ], dtype=np.float32)

    return np.concatenate([feats, extra])


def triangular_membership(x, a, b, c):
    """Funcao fuzzy triangular."""
    x = x.astype(np.float32)
    left = (x - a) / (b - a + 1e-6)
    right = (c - x) / (c - b + 1e-6)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapezoid_membership(x, a, b, c, d):
    """Funcao fuzzy trapezoidal."""
    x = x.astype(np.float32)
    rise = (x - a) / (b - a + 1e-6)
    fall = (d - x) / (d - c + 1e-6)
    return np.clip(np.minimum(np.minimum(rise, 1.0), fall), 0.0, 1.0)


def fuzzy_tone_features(patch, mask=None):
    """
    Mede graus fuzzy para tons diferentes da CT.

    Usa intensidade normalizada 0..255 depois da janela pulmonar.
    Em vez de dizer "preto/branco" duro, cada pixel pode pertencer um pouco a:
      - muito escuro
      - escuro
      - medio
      - claro
      - muito claro

    Isso ajuda quando lesoes aparecem com tons intermediarios ou variaveis.
    """
    if mask is not None and mask.sum() > 0:
        values = patch[mask > 0].astype(np.float32)
    else:
        values = patch.reshape(-1).astype(np.float32)

    if values.size == 0:
        values = patch.reshape(-1).astype(np.float32)

    memberships = {
        "very_dark": trapezoid_membership(values, 0, 0, 35, 75),
        "dark": triangular_membership(values, 35, 85, 135),
        "medium": triangular_membership(values, 95, 145, 195),
        "bright": triangular_membership(values, 155, 205, 245),
        "very_bright": trapezoid_membership(values, 210, 240, 255, 255),
    }

    feats = []
    for degree in memberships.values():
        feats.extend([
            float(degree.mean()),
            float(degree.max()),
            float(np.percentile(degree, 90)),
        ])

    # Entropia fuzzy: alta quando a regiao mistura tons de forma incerta.
    stacked = np.vstack(list(memberships.values())).T
    stacked = stacked / (stacked.sum(axis=1, keepdims=True) + 1e-6)
    fuzzy_entropy = -np.sum(stacked * np.log(stacked + 1e-6), axis=1)
    feats.append(float(fuzzy_entropy.mean()))
    feats.append(float(fuzzy_entropy.std()))

    return np.array(feats, dtype=np.float32)


def spatial_features(patch, mask=None):
    if mask is not None and mask.sum() > 0:
        values = patch[mask > 0].astype(np.float32)
        area = float(mask.sum()) / mask.size
    else:
        values = patch.reshape(-1).astype(np.float32)
        area = 0.0

    if len(values) == 0:
        values = patch.reshape(-1).astype(np.float32)

    hist, _ = np.histogram(values, bins=16, range=(0, 255), density=True)

    # Medidas simples de "branco/preto" e contraste.
    dark_ratio = float((values < 64).mean())
    mid_ratio = float(((values >= 64) & (values < 160)).mean())
    bright_ratio = float((values >= 160).mean())
    p90_p10_contrast = float(np.percentile(values, 90) - np.percentile(values, 10))

    # Borda/aspereza: lesoes podem ter borda e textura diferentes do tecido normal.
    patch_u8 = patch.astype(np.uint8)
    if mask is not None and mask.sum() > 0:
        patch_u8 = (patch_u8 * mask.astype(np.uint8))
    edges = cv2.Canny(patch_u8, 40, 120)
    edge_density = float((edges > 0).mean())

    return np.array([
        values.mean(),
        values.std(),
        np.percentile(values, 10),
        np.percentile(values, 50),
        np.percentile(values, 90),
        area,
        dark_ratio,
        mid_ratio,
        bright_ratio,
        p90_p10_contrast,
        edge_density,
    ] + hist.astype(np.float32).tolist(), dtype=np.float32)


def extract_features(patch, mask=None):
    return np.concatenate([
        spatial_features(patch, mask),
        fuzzy_tone_features(patch, mask),
        fft_radial_features(patch, mask),
    ])


# %%
def build_fourier_dataset(series_zip_dir, rois_by_sop, max_series=None, patch_size=96):
    zip_paths = sorted(series_zip_dir.glob("*.zip"))
    if max_series is not None:
        zip_paths = zip_paths[:max_series]

    rows = []
    X = []
    y = []

    for zip_path in tqdm(zip_paths, desc="Extraindo patches/features"):
        with zipfile.ZipFile(zip_path) as zf:
            dcm_names = sorted(n for n in zf.namelist() if n.lower().endswith(".dcm"))

            for member in dcm_names:
                with zf.open(member) as fh:
                    data = fh.read()

                try:
                    ds = pydicom.dcmread(io.BytesIO(data), force=True)
                    sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
                    series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
                    hu = dicom_to_hu(ds)
                except Exception:
                    continue

                rois = rois_by_sop.get(sop_uid, [])
                img8 = hu_to_uint8_lung(hu)

                if rois:
                    full_mask = mask_from_rois(img8.shape, rois)
                    cx, cy = roi_center(full_mask)
                    patch, patch_mask = crop_centered(img8, full_mask, cx, cy, size=patch_size)
                    features = extract_features(patch, patch_mask)
                    X.append(features)
                    y.append(1)
                    rows.append({
                        "label": "NODULO",
                        "zip": zip_path.name,
                        "member": member,
                        "series_uid": series_uid,
                        "sop_uid": sop_uid,
                        "mask_area": int(full_mask.sum()),
                    })

                    # Salva alguns exemplos para inspecionar visualmente.
                    if len(rows) <= 80:
                        Image.fromarray(patch).save(OUT_DIR / f"exemplo_{len(rows):04d}_NODULO.png")
                        Image.fromarray((patch_mask * 255).astype(np.uint8)).save(
                            OUT_DIR / f"exemplo_{len(rows):04d}_MASK.png"
                        )

                # Amostra negativa sem nodule nesta fatia.
                if random.random() < 0.02:
                    full_mask = np.zeros_like(img8, dtype=np.uint8)
                    neg = random_lung_like_negative(hu, full_mask, size=patch_size)
                    if neg is None:
                        continue
                    patch, patch_mask = neg
                    features = extract_features(patch, patch_mask)
                    X.append(features)
                    y.append(0)
                    rows.append({
                        "label": "SEM_NODULO",
                        "zip": zip_path.name,
                        "member": member,
                        "series_uid": series_uid,
                        "sop_uid": sop_uid,
                        "mask_area": 0,
                    })

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64), pd.DataFrame(rows)


X, y, meta = build_fourier_dataset(
    SERIES_ZIP_DIR,
    rois_by_sop,
    max_series=MAX_SERIES,
    patch_size=PATCH_SIZE,
)

print("X:", X.shape)
print("Distribuicao:", {CLASS_NAMES[k]: v for k, v in Counter(y).items()})
meta.to_csv(OUT_DIR / "amostras_fourier.csv", index=False)


# %%
if len(set(y.tolist())) < 2:
    raise RuntimeError("Ainda nao ha duas classes suficientes. Espere baixar mais series ou aumente MAX_SERIES.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
clf.fit(X_train_s, y_train)

pred = clf.predict(X_test_s)

print(classification_report(y_test, pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", values_format="d")
plt.title("Fourier + segmentacao - LIDC-IDRI")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# %%
feature_names = (
    [
        "mean",
        "std",
        "p10",
        "p50",
        "p90",
        "mask_area_ratio",
        "dark_ratio",
        "mid_ratio",
        "bright_ratio",
        "p90_p10_contrast",
        "edge_density",
    ]
    + [f"hist_{i:02d}" for i in range(16)]
    + [
        f"fuzzy_{name}_{stat}"
        for name in ["very_dark", "dark", "medium", "bright", "very_bright"]
        for stat in ["mean", "max", "p90"]
    ]
    + ["fuzzy_entropy_mean", "fuzzy_entropy_std"]
    + [f"fft_band_{i:02d}" for i in range(24)]
    + ["fft_low", "fft_mid", "fft_high", "fft_high_low_ratio", "fft_mid_low_ratio"]
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(OUT_DIR / "importancia_features_fourier.csv", index=False)
print(importance.head(15))


# %%
# Salva artefatos simples.
import joblib

joblib.dump({
    "scaler": scaler,
    "model": clf,
    "feature_names": feature_names,
    "class_names": CLASS_NAMES,
}, OUT_DIR / "modelo_fourier_segmentacao.joblib")

try:
    def _ensure_models_dir(out_dir):
        models_dir = Path(out_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def _copy_model_to_models_dir(src_path, label, out_dir):
        src = Path(src_path)
        if not src.exists():
            return None
        models_dir = _ensure_models_dir(out_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = models_dir / f"{ts}_{label}{src.suffix}"
        shutil.copy2(src, dest)
        return dest

    _copy_model_to_models_dir(OUT_DIR / "modelo_fourier_segmentacao.joblib", "modelo_fourier_segmentacao", OUT_DIR)
except Exception:
    pass

np.save(OUT_DIR / "X_fourier.npy", X)
np.save(OUT_DIR / "y_fourier.npy", y)

print("Arquivos salvos em:", OUT_DIR)
