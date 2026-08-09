# %% [markdown]
# U-Net para segmentar lesoes do LIDC-IDRI
#
# Objetivo:
# - Usar as mascaras dos XML do LIDC como ground truth.
# - Treinar uma U-Net para aprender a segmentar nodulos.
# - Usar a mascara prevista para reduzir ruido antes da CNN/classificador.
#
# Rode no kernel PyTorch/CUDA. Enquanto o download ainda estiver rodando,
# mantenha MAX_SERIES pequeno. Depois use None.

# %%
# !pip install pydicom tqdm matplotlib pillow opencv-python scikit-learn torch torchvision

import io
import random
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import shutil
import datetime


# %%
LIDC_ROOT = Path("/media/angualberto/C8C814BEC814AD26/TCIA/LIDC-IDRI")
SERIES_ZIP_DIR = LIDC_ROOT / "series_zip"
ANNOTATIONS_ZIP = LIDC_ROOT / "annotations_metadata" / "LIDC-XML-only.zip"
OUT_DIR = LIDC_ROOT / "unet_segmentacao"
OUT_DIR.mkdir(exist_ok=True)

MAX_SERIES = 160
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4


# %%
def strip_ns(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text(node, name):
    for child in node:
        if strip_ns(child.tag) == name:
            return child.text
    return None


def parse_lidc_rois(xml_zip_path):
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
                            points.append((int(float(x)), int(float(y))))

                    if len(points) >= 3:
                        rois_by_sop[sop_uid].append({
                            "series_uid": series_uid,
                            "points": points,
                        })

    return dict(rois_by_sop)


rois_by_sop = parse_lidc_rois(ANNOTATIONS_ZIP)
print("Fatias com mascara:", len(rois_by_sop))


# %%
def dicom_to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def hu_to_uint8_lung(hu, window_min=-1000, window_max=400):
    hu = np.clip(hu, window_min, window_max)
    return ((hu - window_min) / (window_max - window_min) * 255.0).astype(np.uint8)


def mask_from_rois(shape, rois):
    mask = np.zeros(shape, dtype=np.uint8)
    for roi in rois:
        pts = np.array(roi["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def build_positive_index(series_zip_dir, rois_by_sop, max_series=None):
    zip_paths = sorted(series_zip_dir.glob("*.zip"))
    if max_series is not None:
        zip_paths = zip_paths[:max_series]

    items = []
    for zip_path in tqdm(zip_paths, desc="Indexando fatias positivas"):
        with zipfile.ZipFile(zip_path) as zf:
            for member in sorted(n for n in zf.namelist() if n.lower().endswith(".dcm")):
                with zf.open(member) as fh:
                    raw = fh.read()
                try:
                    ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, force=True)
                    sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
                except Exception:
                    continue
                if sop_uid in rois_by_sop:
                    items.append((str(zip_path), member, sop_uid))

    return items


items = build_positive_index(SERIES_ZIP_DIR, rois_by_sop, max_series=MAX_SERIES)
print("Fatias positivas para treino:", len(items))


# %%
class LIDCUNetDataset(Dataset):
    def __init__(self, items, rois_by_sop, img_size=256, augment=False):
        self.items = list(items)
        self.rois_by_sop = rois_by_sop
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        zip_path, member, sop_uid = self.items[idx]
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as fh:
                ds = pydicom.dcmread(io.BytesIO(fh.read()), force=True)

        hu = dicom_to_hu(ds)
        img = hu_to_uint8_lung(hu)
        mask = mask_from_rois(img.shape, self.rois_by_sop[sop_uid])

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.2:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        return torch.from_numpy(img[None, :, :]), torch.from_numpy(mask[None, :, :])


train_items, val_items = train_test_split(items, test_size=0.2, random_state=42)
train_ds = LIDCUNetDataset(train_items, rois_by_sop, img_size=IMG_SIZE, augment=True)
val_ds = LIDCUNetDataset(val_items, rois_by_sop, img_size=IMG_SIZE, augment=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# %%
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
        self.bottleneck = DoubleConv(base * 4, base * 8)

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
        xb = self.bottleneck(F.max_pool2d(x3, 2))

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


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetSmall().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))

best_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for imgs, masks in tqdm(train_dl, desc=f"Treino {epoch:02d}"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = bce(logits, masks) + dice_loss(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            val_loss += loss.item() * imgs.size(0)
            val_dice += dice_score(logits, masks) * imgs.size(0)

    train_loss /= max(1, len(train_ds))
    val_loss /= max(1, len(val_ds))
    val_dice /= max(1, len(val_ds))

    print(f"[{epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), OUT_DIR / "unet_lidc_best.pt")
        print("  salvou melhor modelo")
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

            _copy_model_to_models_dir(OUT_DIR / "unet_lidc_best.pt", "unet_lidc_best", OUT_DIR)
        except Exception:
            pass


# %%
model.load_state_dict(torch.load(OUT_DIR / "unet_lidc_best.pt", map_location=device))
model.eval()

imgs, masks = next(iter(val_dl))
with torch.no_grad():
    logits = model(imgs.to(device)).cpu()
preds = (torch.sigmoid(logits) > 0.5).float()

n = min(4, imgs.size(0))
fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
if n == 1:
    axes = axes[None, :]

for i in range(n):
    axes[i, 0].imshow(imgs[i, 0], cmap="gray")
    axes[i, 0].set_title("CT")
    axes[i, 1].imshow(masks[i, 0], cmap="gray")
    axes[i, 1].set_title("Mascara XML")
    axes[i, 2].imshow(preds[i, 0], cmap="gray")
    axes[i, 2].set_title("U-Net")
    for j in range(3):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "exemplos_unet.png", dpi=150)
plt.show()

print("Melhor Dice:", best_dice)
print("Saida:", OUT_DIR)
