#!/usr/bin/env python3
"""Gera um relatório PDF com métricas de treino encontradas recursivamente.
Procura por arquivos `metrics_by_epoch.csv` sob o diretório raiz fornecido,
plotando curvas e gerando uma página resumo com melhores epochs.
"""
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def find_metrics(root):
    pattern = os.path.join(root, '**', 'metrics_by_epoch.csv')
    return glob.glob(pattern, recursive=True)


def summarize_df(df):
    best_pr = df.loc[df['pr_auc'].idxmax()].to_dict()
    best_roc = df.loc[df['roc_auc'].idxmax()].to_dict()
    best_loss = df.loc[df['loss'].idxmin()].to_dict()
    return {'best_pr': best_pr, 'best_roc': best_roc, 'best_loss': best_loss}


def plot_run(axs, df, label):
    ax_loss, ax_auc, ax_pr, ax_acc = axs
    epochs = df['epoch']
    if 'loss' in df:
        ax_loss.plot(epochs, df['loss'], label=label)
    if 'roc_auc' in df:
        ax_auc.plot(epochs, df['roc_auc'], label=label)
    if 'pr_auc' in df:
        ax_pr.plot(epochs, df['pr_auc'], label=label)
    if 'accuracy' in df:
        ax_acc.plot(epochs, df['accuracy'], label=label)


def make_table_page(pdf, summaries):
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.axis('off')
    rows = []
    for name,s in summaries.items():
        br = s['best_pr']
        rows.append([name, int(br.get('epoch',0)), round(br.get('pr_auc',0),4), round(br.get('roc_auc',0),4), round(br.get('loss',0),6)])
    collabels = ['run','best_pr_epoch','pr_auc','roc_auc','loss']
    table = ax.table(cellText=rows, colLabels=collabels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1,2)
    ax.set_title('Resumo: melhores epochs por run')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def generate_report(root, out_pdf):
    files = find_metrics(root)
    if not files:
        print('Nenhum metrics_by_epoch.csv encontrado em', root)
        return 1
    summaries = {}
    combined = {}
    # prepare combined plots
    fig, axs_grid = plt.subplots(2,2,figsize=(11.69,8.27))
    ax_loss = axs_grid[0,0]
    ax_auc = axs_grid[0,1]
    ax_pr = axs_grid[1,0]
    ax_acc = axs_grid[1,1]
    axs = (ax_loss, ax_auc, ax_pr, ax_acc)

    with PdfPages(out_pdf) as pdf:
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print('Falha ao ler', f, e)
                continue
            run_name = os.path.basename(os.path.dirname(f))
            summaries[run_name] = summarize_df(df)
            plot_run(axs, df, run_name)

            # page per run: plots
            fig_run, (a1,a2) = plt.subplots(1,2,figsize=(11.69,4))
            if 'loss' in df:
                a1.plot(df['epoch'], df['loss'], marker='o')
                a1.set_title(f'{run_name} - loss')
                a1.set_xlabel('epoch')
            if 'roc_auc' in df:
                a2.plot(df['epoch'], df['roc_auc'], marker='o', label='roc_auc')
            if 'pr_auc' in df:
                a2.plot(df['epoch'], df['pr_auc'], marker='x', label='pr_auc')
            a2.set_title(f'{run_name} - aucs')
            a2.set_xlabel('epoch')
            a2.legend()
            pdf.savefig(fig_run, bbox_inches='tight')
            plt.close(fig_run)

        # combined page
        ax_auc.set_title('ROC AUC por epoch (comparativo)')
        ax_auc.set_xlabel('epoch')
        ax_pr.set_title('PR AUC por epoch (comparativo)')
        ax_pr.set_xlabel('epoch')
        ax_loss.set_title('Loss por epoch (comparativo)')
        ax_loss.set_xlabel('epoch')
        ax_acc.set_title('Accuracy por epoch (comparativo)')
        ax_acc.set_xlabel('epoch')
        ax_auc.legend()
        ax_pr.legend()
        ax_loss.legend()
        ax_acc.legend()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # summary table
        make_table_page(pdf, summaries)

    print('Relatório gerado em', out_pdf)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Diretório raiz para procurar métricas')
    p.add_argument('--out', required=True, help='Caminho do PDF de saída')
    args = p.parse_args()
    exit(generate_report(args.root, args.out))
