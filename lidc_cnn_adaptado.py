# %% [markdown]
# CNN adaptada para TCIA LIDC-IDRI
#
# Este arquivo substitui o dataset antigo de JPG/PNG por fatias DICOM do
# LIDC-IDRI baixadas no HD externo. As classes passam a ser:
# - 0: SEM_NODULO
# - 1: NODULO
#
# Abra este arquivo no VS Code/Jupyter e execute célula por célula.

# %%
# Se faltar dependência no kernel do notebook, rode esta célula uma vez:
# !pip install pydicom pandas scikit-learn tqdm matplotlib pillow opencv-python torch torchvision

import io
import os
import random
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from tqdm import tqdm

import matplotlib.pyplot as plt
import shutil
import datetime


# %%
# Caminhos do dataset no HD de 1 TB.
LIDC_ROOT = Path("/media/angualberto/C8C814BEC814AD26/TCIA/LIDC-IDRI")
SERIES_ZIP_DIR = LIDC_ROOT / "series_zip"
ANNOTATIONS_ZIP = LIDC_ROOT / "annotations_metadata" / "LIDC-XML-only.zip"

# Enquanto o download ainda estiver rodando, use um valor pequeno.
# Depois que terminar, coloque None para usar todas as series.
MAX_SERIES = 80

CLASS_NAMES = ["SEM_NODULO", "NODULO"]

print("Series ZIP:", SERIES_ZIP_DIR)
print("XML:", ANNOTATIONS_ZIP)
print("ZIPs baixados agora:", len(list(SERIES_ZIP_DIR.glob("*.zip"))))


# %%
def strip_namespace(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text(node, name):
    for child in node:
        if strip_namespace(child.tag) == name:
            return child.text
    return None


def parse_lidc_positive_sops(xml_zip_path):
    """
    Retorna:
      dict[SeriesInstanceUID] -> set[SOPInstanceUID anotado como nódulo]

    Cada XML do LIDC contém ROIs com imageSOP_UID. Se uma fatia DICOM tem esse
    SOPInstanceUID, ela recebe label NODULO.
    """
    positives = defaultdict(set)

    with zipfile.ZipFile(xml_zip_path) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]

        for name in tqdm(xml_names, desc="Lendo XML LIDC"):
            try:
                root = ET.fromstring(zf.read(name))
            except ET.ParseError:
                continue

            series_uid = None
            for node in root.iter():
                if strip_namespace(node.tag) == "SeriesInstanceUid":
                    series_uid = node.text
                    break
            if not series_uid:
                continue

            for node in root.iter():
                if strip_namespace(node.tag) == "roi":
                    sop_uid = child_text(node, "imageSOP_UID")
                    inclusion = child_text(node, "inclusion")
                    if sop_uid and str(inclusion).upper() == "TRUE":
                        positives[series_uid].add(sop_uid)

    return dict(positives)


positive_sops_by_series = parse_lidc_positive_sops(ANNOTATIONS_ZIP)
print("Series com anotacao:", len(positive_sops_by_series))
print("Fatias positivas anotadas:", sum(len(v) for v in positive_sops_by_series.values()))


# %%
def dicom_to_pil(ds, window_min=-1000, window_max=400):
    """
    Converte uma fatia DICOM CT para PIL RGB.
    Usa janela pulmonar simples: HU [-1000, 400].
    """
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept

    hu = np.clip(hu, window_min, window_max)
    img = ((hu - window_min) / (window_max - window_min) * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


def list_dicom_members(zf):
    return sorted(
        [n for n in zf.namelist() if n.lower().endswith(".dcm")],
        key=lambda x: x.lower(),
    )


def build_lidc_index(series_zip_dir, positive_map, max_series=None, neg_per_pos=2):
    """
    Monta um indice de fatias:
      (zip_path, member_name, label, series_uid, sop_uid)

    Para evitar um dataset gigante e desbalanceado, mantém todas as fatias
    positivas e amostra algumas negativas por serie.
    """
    zip_paths = sorted(series_zip_dir.glob("*.zip"))
    if max_series is not None:
        zip_paths = zip_paths[:max_series]

    samples = []

    for zip_path in tqdm(zip_paths, desc="Indexando ZIPs DICOM"):
        with zipfile.ZipFile(zip_path) as zf:
            members = list_dicom_members(zf)
            if not members:
                continue

            pos_items = []
            neg_items = []
            series_uid = None

            for member in members:
                with zf.open(member) as fh:
                    data = fh.read()
                try:
                    ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
                except Exception:
                    continue

                series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
                sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
                label = int(sop_uid in positive_map.get(series_uid, set()))
                item = (str(zip_path), member, label, series_uid, sop_uid)
                if label:
                    pos_items.append(item)
                else:
                    neg_items.append(item)

            if pos_items:
                samples.extend(pos_items)
                keep_neg = min(len(neg_items), max(1, len(pos_items) * neg_per_pos))
                samples.extend(random.sample(neg_items, keep_neg))
            else:
                keep_neg = min(len(neg_items), 3)
                samples.extend(random.sample(neg_items, keep_neg))

    random.shuffle(samples)
    return samples


samples = build_lidc_index(
    SERIES_ZIP_DIR,
    positive_sops_by_series,
    max_series=MAX_SERIES,
    neg_per_pos=2,
)

labels = [s[2] for s in samples]
print("Total de fatias indexadas:", len(samples))
print("Distribuicao:", {CLASS_NAMES[k]: v for k, v in Counter(labels).items()})


# %%
class LIDCSliceDataset(Dataset):
    def __init__(self, samples, transform=None, mode="train"):
        self.samples = list(samples)
        self.transform = transform
        self.mode = mode
        self.class_names = CLASS_NAMES
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.class_tags = [CLASS_NAMES[s[2]] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        zip_path, member, label, series_uid, sop_uid = self.samples[idx]
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as fh:
                ds = pydicom.dcmread(io.BytesIO(fh.read()), force=True)

        img = dicom_to_pil(ds)

        if self.transform:
            img = self.transform(img)

        meta = {
            "zip": Path(zip_path).name,
            "member": member,
            "series_uid": series_uid,
            "sop_uid": sop_uid,
        }
        return img, int(label)


# %%
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# %%
train_indices, test_indices = train_test_split(
    range(len(samples)),
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

dataset_train = LIDCSliceDataset([samples[i] for i in train_indices], transform=train_transform)
dataset_test = LIDCSliceDataset([samples[i] for i in test_indices], transform=test_transform)

loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=2)
loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=2)

print("Treino:", len(dataset_train), Counter(dataset_train.class_tags))
print("Teste:", len(dataset_test), Counter(dataset_test.class_tags))


# %%
class SimpleEnsemble:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        self.model1 = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.model1.aux_logits = False
        self.model1.fc = nn.Identity()

        self.model2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model2.fc = nn.Identity()

        for model in [self.model1, self.model2]:
            model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def extract_features(self, x):
        x = x.to(self.device)
        feat1 = self.model1(x)
        feat2 = self.model2(x)
        return torch.cat([feat1, feat2], dim=1)


def extract_features_smart(dataset, feature_extractor, batch_size=8, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extraindo features"):
            feats = feature_extractor.extract_features(imgs)
            features_list.append(feats.cpu())
            labels_list.append(lbls.cpu())

    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando device:", device)

feature_extractor = SimpleEnsemble(device=device)
features_train, labels_train = extract_features_smart(dataset_train, feature_extractor, batch_size=8)
features_test, labels_test = extract_features_smart(dataset_test, feature_extractor, batch_size=8)

print("features_train:", features_train.shape)
print("features_test:", features_test.shape)


# %%
X_train = features_train.numpy()
y_train_np = labels_train.numpy()
X_test = features_test.numpy()
y_test_np = labels_test.numpy()

n_pca = min(100, X_train.shape[0] - 1, X_train.shape[1])
pca = PCA(n_components=n_pca, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

try:
    lda = LDA(n_components=min(len(CLASS_NAMES) - 1, n_pca))
    Z_train = lda.fit_transform(X_train_pca, y_train_np)
    Z_test = lda.transform(X_test_pca)
except Exception as e:
    print("Erro no LDA, usando PCA:", e)
    Z_train = X_train_pca[:, :min(10, n_pca)]
    Z_test = X_test_pca[:, :min(10, n_pca)]

Z_train = torch.from_numpy(Z_train).float()
Z_test = torch.from_numpy(Z_test).float()
y_train = torch.from_numpy(y_train_np).long()
y_test = torch.from_numpy(y_test_np).long()

print("Z_train:", Z_train.shape)


# %%
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = FeatureDataset(Z_train, y_train)
test_ds = FeatureDataset(Z_test, y_test)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)


# %%
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(Z_train.shape[1], len(CLASS_NAMES)).to(device)

class_counts = Counter(y_train.tolist())
weights = torch.tensor(
    [1.0 / max(1, class_counts[i]) for i in range(len(CLASS_NAMES))],
    dtype=torch.float32,
).to(device)
weights = weights / weights.sum() * len(CLASS_NAMES)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Modelo no device:", device)
print("Input dim:", Z_train.shape[1], "| Classes:", CLASS_NAMES)


# %%
for epoch in range(25):
    model.train()
    total_loss = 0.0

    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_ds)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    acc = correct / max(1, total)
    print(f"[{epoch + 1:02d}] loss={avg_loss:.5f} | acc={acc * 100:.2f}%")


# %%
model.eval()
preds = []
reals = []

with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu()
        preds.append(pred)
        reals.append(yb.cpu())

preds = torch.cat(preds)
reals = torch.cat(reals)

cm = confusion_matrix(reals, preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusao - LIDC-IDRI")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

acc = 100 * (cm.diagonal().sum() / max(1, cm.sum()))
f1_macro = f1_score(reals, preds, average="macro") * 100
f1_weighted = f1_score(reals, preds, average="weighted") * 100
f1_per_class = f1_score(reals, preds, average=None, labels=[0, 1]) * 100

print(f"Acuracia total: {acc:.2f}%")
print(f"F1 macro: {f1_macro:.2f}%")
print(f"F1 weighted: {f1_weighted:.2f}%")
for name, score in zip(CLASS_NAMES, f1_per_class):
    print(f"{name}: {score:.2f}%")


# %%
# Salvar artefatos para reutilizar depois.
OUT_DIR = LIDC_ROOT / "modelo_lidc"
OUT_DIR.mkdir(exist_ok=True)

torch.save(model.state_dict(), OUT_DIR / "classifier_lidc.pt")
torch.save(
    {
        "class_names": CLASS_NAMES,
        "train_samples": [samples[i] for i in train_indices],
        "test_samples": [samples[i] for i in test_indices],
    },
    OUT_DIR / "lidc_split_metadata.pt",
)

pd.DataFrame({
    "real": [CLASS_NAMES[i] for i in reals.tolist()],
    "pred": [CLASS_NAMES[i] for i in preds.tolist()],
}).to_csv(OUT_DIR / "predicoes_teste_lidc.csv", index=False)

print("Salvo em:", OUT_DIR)
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

    _copy_model_to_models_dir(OUT_DIR / "classifier_lidc.pt", "classifier_lidc", OUT_DIR)
    _copy_model_to_models_dir(OUT_DIR / "lidc_split_metadata.pt", "lidc_split_metadata", OUT_DIR)
except Exception:
    pass
