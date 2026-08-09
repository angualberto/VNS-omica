#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from .spectral_attention import SpectralAttention


def load_color_vectors(colors_csv):
    df = pd.read_csv(colors_csv)
    # standardize columns
    if 'hue_medio_graus' in df.columns and 'hue_medio' not in df.columns:
        df = df.rename(columns={'hue_medio_graus': 'hue_medio'})
    freq_cols = [c for c in df.columns if c.startswith('freq_')]
    extra = []
    if 'hue_medio' in df.columns:
        extra.append('hue_medio')
    cols = freq_cols + extra
    cmap = {}
    for _, row in df.iterrows():
        basename = os.path.splitext(os.path.basename(row.get('image_path', '')))[0] if 'image_path' in df.columns else ''
        vals = [float(row.get(c, 0.0)) if pd.notna(row.get(c)) else 0.0 for c in cols]
        if basename:
            cmap[basename] = np.array(vals, dtype=np.float32)
        if 'image_id' in df.columns and pd.notna(row['image_id']):
            cmap[str(row['image_id'])] = np.array(vals, dtype=np.float32)
    return cmap, cols

class SimpleVecDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def pretrain(colors_csv, cases_csv, out_model, epochs=3, batch_size=64, device='cpu'):
    cmap, cols = load_color_vectors(colors_csv)
    df = pd.read_csv(cases_csv)
    # prefer image_id, else basename
    X = []
    Y = []
    for _, row in df.iterrows():
        key = None
        if 'image_id' in row.index and pd.notna(row.get('image_id')):
            key = str(row['image_id'])
        else:
            key = os.path.splitext(os.path.basename(row['image_path']))[0]
        vec = cmap.get(key)
        if vec is None:
            continue
        X.append(vec)
        y = row.get('label', None)
        if y is None and 'cancer' in row.index:
            y = row['cancer']
        if y is None:
            continue
        Y.append(int(y))
    if len(Y) == 0:
        raise RuntimeError('No matching color vectors found for provided cases CSV')
    X = np.stack(X)
    Y = np.array(Y)
    pos = max(1, int(Y.sum()))
    neg = max(1, int((Y == 0).sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    ds = SimpleVecDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = SpectralAttention(input_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(ds)
        print(f"[pretrain] epoch {ep} loss={avg:.4f}")
    os.makedirs(os.path.dirname(out_model) or '.', exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'cols': cols}, out_model)
    print('Saved pretrain model to', out_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--colors-csv', required=True)
    parser.add_argument('--cases-csv', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    pretrain(args.colors_csv, args.cases_csv, args.out_model, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
