# -*- coding: utf-8 -*-
"""
06_model_cnn.py

This is my first deep learning model, a small CNN I built from scratch
rather than a pretrained backbone. I wanted a from-scratch baseline in
the mix so the comparison later actually shows what transfer learning
(ResNet50, EfficientNetB0) buys me over training on CK+ alone.

Four conv blocks with batch norm and increasing channel depth
(32 -> 64 -> 128 -> 256), then an adaptive pool so the classifier head
doesn't care about small changes in input resolution, followed by a
dropout layer before the final linear output. Nothing exotic, just a
solid small-dataset CNN.

Run this after 02_preprocess_data.py:
    python src/06_model_cnn.py
"""

import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANIFEST_CSV, MODELS_DIR, FIGURES_DIR, EMOTION_LABELS, ensure_dirs
from utils import (set_seed, get_device, FERDataset, get_train_transform,
                    get_eval_transform, train_deep_model, evaluate_deep_model,
                    plot_training_curves, plot_confusion_matrix, save_metrics_row)

MODEL_NAME = 'CNN'
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 32
METRICS_CSV = os.path.join(os.path.dirname(FIGURES_DIR), 'metrics_summary.csv')


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_loaders(df):
    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']
    test_df  = df[df['split'] == 'test']

    train_ds = FERDataset(train_df['path'].tolist(), train_df['label_idx'].tolist(), get_train_transform())
    val_ds   = FERDataset(val_df['path'].tolist(),   val_df['label_idx'].tolist(),   get_eval_transform())
    test_ds  = FERDataset(test_df['path'].tolist(),  test_df['label_idx'].tolist(),  get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def main():
    ensure_dirs()
    set_seed()
    device = get_device()

    if not os.path.exists(MANIFEST_CSV):
        print('manifest.csv not found, run 02_preprocess_data.py first.')
        return

    df = pd.read_csv(MANIFEST_CSV)
    train_loader, val_loader, test_loader = build_loaders(df)
    print(f'Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}')

    model = SimpleCNN(num_classes=len(EMOTION_LABELS))
    model, history = train_deep_model(
        model, MODEL_NAME, train_loader, val_loader, device, MODELS_DIR,
        epochs=EPOCHS, lr=LR
    )

    acc, preds, trues = evaluate_deep_model(model, MODEL_NAME, test_loader, device, EMOTION_LABELS)

    plot_training_curves(
        history, MODEL_NAME, '#2ecc71',
        os.path.join(FIGURES_DIR, 'cnn_training_curves.png')
    )
    plot_confusion_matrix(
        trues, preds, EMOTION_LABELS, MODEL_NAME, acc,
        os.path.join(FIGURES_DIR, 'cnn_confusion_matrix.png')
    )
    save_metrics_row(METRICS_CSV, MODEL_NAME, 'Deep Learning', trues, preds)


if __name__ == '__main__':
    main()
