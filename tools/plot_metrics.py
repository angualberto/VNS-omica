#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
else:
    csv_path = Path(__file__).resolve().parent.parent / 'treino_calc_fuzzy_spectral_avl' / 'metrics_by_epoch.csv'

if not csv_path.exists():
    print('CSV not found:', csv_path)
    sys.exit(1)

out_dir = csv_path.parent

df = pd.read_csv(csv_path)
if 'epoch' in df.columns:
    df['epoch'] = df['epoch'].astype(int)
    df = df.sort_values('epoch')

# Metrics plot
metrics = ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1']
try:
    plt.style.use('seaborn-darkgrid')
except Exception:
    plt.style.use('ggplot')
plt.figure(figsize=(10,6))
for m in metrics:
    if m in df.columns:
        plt.plot(df['epoch'], df[m], marker='o', label=m)
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('PR/ROC/Precision/Recall/F1 por Epoch')
plt.legend()
plt.tight_layout()
metrics_img = out_dir / 'metrics_over_epochs.png'
plt.savefig(metrics_img)
plt.close()

# Loss plot
plt.figure(figsize=(8,5))
if 'loss' in df.columns:
    plt.plot(df['epoch'], df['loss'], marker='o', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss por Epoch')
    plt.tight_layout()
    loss_img = out_dir / 'loss_over_epochs.png'
    plt.savefig(loss_img)
    plt.close()
else:
    loss_img = None

# Accuracy plot
if 'accuracy' in df.columns:
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['accuracy'], marker='o', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy por Epoch')
    plt.tight_layout()
    acc_img = out_dir / 'accuracy_over_epochs.png'
    plt.savefig(acc_img)
    plt.close()
else:
    acc_img = None

print('Saved:', metrics_img)
if loss_img:
    print('Saved:', loss_img)
if acc_img:
    print('Saved:', acc_img)
