from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

try:
    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import disk
except Exception:  # pragma: no cover
    rank_entropy = None
    disk = None

EPS = 1e-6


def _safe01(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo + EPS:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def _first_numeric(value) -> float:
    """Return the first numeric DICOM value, including MultiValue fields."""
    if pydicom is not None and isinstance(value, pydicom.multival.MultiValue):
        value = value[0]
    return float(value)


def read_dicom_gray_with_metadata(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    """Read DICOM mammogram and return normalized pixels plus preprocessing metadata."""
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM mammograms")
    ds = pydicom.dcmread(str(path), force=True)
    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    inverted = photometric == "MONOCHROME1"
    if inverted:
        img = img.max() - img

    wc = None
    ww = None
    windowing_applied = False
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = _first_numeric(ds.WindowCenter)
        ww = _first_numeric(ds.WindowWidth)
        if ww > 0:
            img = np.clip(img, wc - ww / 2.0, wc + ww / 2.0)
            windowing_applied = True

    metadata = {
        "photometric_interpretation": photometric,
        "monochrome1_inverted": inverted,
        "window_center": wc,
        "window_width": ww,
        "windowing_applied": windowing_applied,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "rows": int(getattr(ds, "Rows", img.shape[0])),
        "columns": int(getattr(ds, "Columns", img.shape[1])),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
    }
    return _safe01(img), metadata


def read_dicom_gray(path: str | Path) -> np.ndarray:
    img, _metadata = read_dicom_gray_with_metadata(path)
    return img


def load_mammo_image(path: str | Path) -> np.ndarray:
    """Load DICOM or common grayscale image as float32 in [0, 1]."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom", ""}:
        return read_dicom_gray(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return _safe01(img)


def resize_or_pad(img: np.ndarray, size: int = 256) -> np.ndarray:
    if img.shape == (size, size):
        return img.astype(np.float32)
    return cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    return _safe01(patch)


def sobel_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    return _safe01(sx), _safe01(sy), _safe01(mag)


def local_entropy_map(img: np.ndarray, radius: int = 5) -> np.ndarray:
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if rank_entropy is not None and disk is not None:
        ent = rank_entropy(img_u8, disk(radius)).astype(np.float32)
        return _safe01(ent)
    # Fallback: local variance as a cheap entropy proxy.
    mean = cv2.blur(img.astype(np.float32), (radius * 2 + 1, radius * 2 + 1))
    mean2 = cv2.blur((img.astype(np.float32) ** 2), (radius * 2 + 1, radius * 2 + 1))
    var = np.maximum(mean2 - mean**2, 0.0)
    return _safe01(var)


def fft_low_high_maps(img: np.ndarray, low_radius: float = 0.12, high_radius: float = 0.32) -> tuple[np.ndarray, np.ndarray]:
    """Approximate low/high frequency reconstructions as spatial maps."""
    arr = img.astype(np.float32)
    h, w = arr.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    rr = rr / (rr.max() + EPS)
    f = np.fft.fftshift(np.fft.fft2(arr))
    low_mask = rr <= low_radius
    high_mask = rr >= high_radius
    low = np.real(np.fft.ifft2(np.fft.ifftshift(f * low_mask)))
    high = np.real(np.fft.ifft2(np.fft.ifftshift(f * high_mask)))
    return _safe01(low), _safe01(np.abs(high))


def pseudo_color_family_map(img: np.ndarray, grad_mag: np.ndarray | None = None) -> np.ndarray:
    """Simplified AlizaMS/RainbowB-like color-family scalar channel.

    0.25 = roxo/azulado, 0.50 = esverdeado, 0.75 = amarelado/marrom, 1.00 = vermelho/rosado.
    Intensity and gradient are mixed so strong bright edges move toward warm families.
    """
    if grad_mag is None:
        _, _, grad_mag = sobel_channels(img)
    score = np.clip(0.78 * img + 0.22 * grad_mag, 0.0, 1.0)
    out = np.zeros_like(score, dtype=np.float32)
    out[(score >= 0.00) & (score < 0.35)] = 0.25
    out[(score >= 0.35) & (score < 0.62)] = 0.50
    out[(score >= 0.62) & (score < 0.82)] = 0.75
    out[score >= 0.82] = 1.00
    return out


def make_8ch_features(patch: np.ndarray, patch_size: int = 256) -> torch.Tensor:
    """Return tensor [8, patch_size, patch_size] from a grayscale patch/image."""
    img = normalize_patch(resize_or_pad(patch, patch_size))
    sx, sy, mag = sobel_channels(img)
    ent = local_entropy_map(img)
    fft_low, fft_high = fft_low_high_maps(img)
    pseudo = pseudo_color_family_map(img, mag)
    stack = np.stack([img, mag, sx, sy, ent, fft_low, fft_high, pseudo], axis=0)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.from_numpy(stack)
