# -*- coding: utf-8 -*-
"""
09_compare_models.py

This is the last step. By the time I get here, every one of the five
model scripts has already appended its row to metrics_summary.csv, so
this script's job is just to read that file and turn it into the
comparison figures and table I actually want to look at, side by side
bar charts, a grouped metrics chart, and a plain CSV table.

I made this its own script instead of folding it into each model file so
I can regenerate the comparison any time without retraining anything, as
long as metrics_summary.csv already has all five rows in it.

Run this after all five model scripts have been run at least once:
    python src/09_compare_models.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, BASE_DIR, MODEL_COLORS, ACCURACY_TARGET, ensure_dirs
from utils import apply_plot_style

METRICS_CSV = os.path.join(BASE_DIR, 'metrics_summary.csv')
EXPECTED_MODELS = ['KNN', 'SVM', 'CNN', 'ResNet50', 'EfficientNetB0']


def load_metrics():
    if not os.path.exists(METRICS_CSV):
        print(f'{METRICS_CSV} not found. Run the five model scripts first,')
        print('each one appends its own row to this file when it finishes.')
        return None

    df = pd.read_csv(METRICS_CSV)
    missing = [m for m in EXPECTED_MODELS if m not in df['Model'].tolist()]
    if missing:
        print(f'Note: I do not have results yet for: {missing}')
        print('The comparison below only covers whichever models have already run.')

    # I keep a fixed model order everywhere so every chart lines up the
    # same way instead of following whatever order the CSV rows landed in
    order = [m for m in EXPECTED_MODELS if m in df['Model'].tolist()]
    df['Model'] = pd.Categorical(df['Model'], categories=order, ordered=True)
    return df.sort_values('Model').reset_index(drop=True)


def plot_accuracy_bar(df, save_path):
    apply_plot_style()
    colors = [MODEL_COLORS.get(m, '#34495e') for m in df['Model']]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(df['Model'].astype(str), df['Accuracy'], color=colors,
                   edgecolor='black', linewidth=0.8, width=0.55)
    ax.axhline(ACCURACY_TARGET, color='red', linestyle='--', lw=1.6,
               label=f'{ACCURACY_TARGET}% Reference Line')
    ax.set_ylim(0, 108)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Model Comparison — Facial Expression Recognition (CK+)',
                  fontsize=13, fontweight='bold')

    for bar, acc in zip(bars, df['Accuracy']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_grouped_metrics(df, save_path):
    apply_plot_style()
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metric_names))
    n_models = len(df)
    width = 0.8 / max(n_models, 1)
    offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(13, 6))
    for (_, row), offset in zip(df.iterrows(), offsets):
        vals = [row[m] for m in metric_names]
        color = MODEL_COLORS.get(row['Model'], '#34495e')
        ax.bar(x + offset, vals, width, label=str(row['Model']),
               color=color, edgecolor='black', linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.axhline(ACCURACY_TARGET, color='red', linestyle='--', lw=1.5,
               label=f'{ACCURACY_TARGET}% Reference')
    ax.set_title('Performance Metrics Comparison — All Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {save_path}')


def print_summary_table(df):
    print('\n' + '=' * 72)
    print('Performance Metrics Summary — All Models')
    print('=' * 72)
    display_df = df[['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        display_df[col] = display_df[col].map(lambda v: f'{v:.2f}%')
    print(display_df.to_string(index=False))
    print('=' * 72)


def main():
    ensure_dirs()
    df = load_metrics()
    if df is None or len(df) == 0:
        return

    print_summary_table(df)

    plot_accuracy_bar(df, os.path.join(FIGURES_DIR, 'comparison_accuracy_bar.png'))
    plot_grouped_metrics(df, os.path.join(FIGURES_DIR, 'comparison_metrics_grouped.png'))

    final_table_path = os.path.join(FIGURES_DIR, 'comparison_table.csv')
    df.to_csv(final_table_path, index=False)
    print(f'\nFinal comparison table saved to {final_table_path}')
    print('Comparison complete.')


if __name__ == '__main__':
    main()
