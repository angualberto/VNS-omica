#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem')
FEATURE_DIR = ROOT / 'sistema_integrado_rsna_csv_treino' / 'classificador_patches_xgboost_wavelet_gradiente_cor_fft'
CSV_PATH = FEATURE_DIR / 'features_imagem_patches.csv'
OUT_DIR = FEATURE_DIR / 'vnn_volterra'
TARGET_COL = 'cancer'

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
PRED_THRESHOLD = 0.45
RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MammoFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class VolterraLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(in_features, rank, bias=False)
        self.quad = nn.Linear(rank, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(self, x):
        linear_term = self.linear(x)
        u = self.U(x)
        v = self.V(x)
        quadratic_term = self.quad(u * v)
        out = linear_term + quadratic_term
        out = self.norm(out)
        return self.act(out)


class VNNClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.v1 = VolterraLayer(input_dim, 128, rank=64)
        self.drop1 = nn.Dropout(0.30)
        self.v2 = VolterraLayer(128, 64, rank=32)
        self.drop2 = nn.Dropout(0.30)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.v1(x)
        x = self.drop1(x)
        x = self.v2(x)
        x = self.drop2(x)
        return self.out(x)


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(CSV_PATH)
    if 'split' not in df.columns:
        raise RuntimeError('O CSV precisa da coluna split para usar o split train/test ja definido.')

    meta_cols = {'split', 'patient_id', 'image_id', 'image_path', TARGET_COL}
    numeric_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError('Split train/test vazio no CSV.')

    X_train = train_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X_test = test_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    y_train = train_df[TARGET_COL].to_numpy(np.float32)
    y_test = test_df[TARGET_COL].to_numpy(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test, numeric_cols, scaler, train_df, test_df


def evaluate(model: nn.Module, loader: DataLoader, y_true: np.ndarray) -> tuple[np.ndarray, dict]:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
    probs_arr = np.asarray(probs, dtype=np.float32)
    preds = (probs_arr >= PRED_THRESHOLD).astype(int)
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, probs_arr)),
        'pr_auc': float(average_precision_score(y_true, probs_arr)),
        'confusion_matrix': confusion_matrix(y_true, preds).tolist(),
        'classification_report': classification_report(y_true, preds, output_dict=True, zero_division=0),
    }
    return probs_arr, metrics


def save_curves(y_true: np.ndarray, probs: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    roc = roc_auc_score(y_true, probs)
    pr = average_precision_score(y_true, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'VNN AUC={roc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Falso positivo')
    plt.ylabel('Verdadeiro positivo')
    plt.title('ROC - VNN Volterra')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'roc_vnn.png', dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f'VNN PR={pr:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precisao')
    plt.title('Precision-Recall - VNN Volterra')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pr_vnn.png', dpi=160)
    plt.close()


def main() -> None:
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Device:', DEVICE)
    print('CSV:', CSV_PATH)

    X_train, X_test, y_train, y_test, feature_names, scaler, train_df, test_df = prepare_data()
    print(f'Treino: {len(y_train)} imagens | Teste: {len(y_test)} imagens | Features: {len(feature_names)}')
    print(f'Treino cancer={int(y_train.sum())} sem_cancer={int(len(y_train)-y_train.sum())}')
    print(f'Teste cancer={int(y_test.sum())} sem_cancer={int(len(y_test)-y_test.sum())}')

    train_loader = DataLoader(MammoFeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MammoFeatureDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = VNNClassifier(input_dim=X_train.shape[1]).to(DEVICE)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    history = []
    best_pr = -1.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
        total_loss /= len(train_loader.dataset)

        probs, metrics = evaluate(model, test_loader, y_test)
        history.append({'epoch': epoch + 1, 'loss': total_loss, **{k: v for k, v in metrics.items() if k in {'roc_auc', 'pr_auc'}}})
        if metrics['pr_auc'] > best_pr:
            best_pr = metrics['pr_auc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:03d}/{EPOCHS} loss={total_loss:.4f} ROC={metrics["roc_auc"]:.4f} PR={metrics["pr_auc"]:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    probs, metrics = evaluate(model, test_loader, y_test)
    preds = (probs >= PRED_THRESHOLD).astype(int)

    pd.DataFrame(history).to_csv(OUT_DIR / 'historico_treino_vnn.csv', index=False)
    pred_df = test_df[['patient_id', 'image_id', 'image_path', TARGET_COL]].copy()
    pred_df['prob_cancer_vnn'] = probs
    pred_df['pred_vnn'] = preds
    pred_df.to_csv(OUT_DIR / 'predicoes_vnn.csv', index=False)
    save_curves(y_test, probs)

    torch.save(model.state_dict(), OUT_DIR / 'modelo_vnn_volterra.pt')
    with (OUT_DIR / 'scaler_feature_names.pkl').open('wb') as f:
        pickle.dump({'scaler': scaler, 'feature_names': feature_names}, f)

    payload = {
        'device': DEVICE,
        'csv_path': str(CSV_PATH),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'prediction_threshold': PRED_THRESHOLD,
        'train_n': int(len(y_train)),
        'test_n': int(len(y_test)),
        'feature_count': int(len(feature_names)),
        'best_pr_auc_during_training': float(best_pr),
        **metrics,
    }
    (OUT_DIR / 'metricas_vnn.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    report = metrics['classification_report']
    lines = [
        'RELATORIO VNN VOLTERRA - MAMOGRAFIA FEATURES POR PATCH\n',
        f'Device: {DEVICE}\n',
        f'Treino: {len(y_train)} | Teste: {len(y_test)} | Features: {len(feature_names)}\n',
        f'Limiar de predicao: {PRED_THRESHOLD:.2f}\n',
        f'ROC AUC: {metrics["roc_auc"]:.4f}\n',
        f'PR AUC: {metrics["pr_auc"]:.4f}\n',
        f'Matriz confusao [[TN, FP], [FN, TP]]: {metrics["confusion_matrix"]}\n',
    ]
    if '1.0' in report:
        c = report['1.0']
    else:
        c = report.get('1', {})
    if c:
        lines.append(f'Cancer precision: {c["precision"]:.4f} | recall: {c["recall"]:.4f} | f1: {c["f1-score"]:.4f}\n')
    lines.append('\nArquivos:\n')
    for name in ['modelo_vnn_volterra.pt', 'predicoes_vnn.csv', 'historico_treino_vnn.csv', 'roc_vnn.png', 'pr_vnn.png', 'metricas_vnn.json']:
        lines.append(str(OUT_DIR / name) + '\n')
    (OUT_DIR / 'relatorio_vnn.txt').write_text(''.join(lines), encoding='utf-8')

    print('\nRESULTADOS VNN')
    print('ROC AUC:', metrics['roc_auc'])
    print('PR AUC:', metrics['pr_auc'])
    print('Matriz confusao:')
    print(np.asarray(metrics['confusion_matrix']))
    print(classification_report(y_test, preds, digits=4, zero_division=0))
    print('Saida:', OUT_DIR)


if __name__ == '__main__':
    main()
