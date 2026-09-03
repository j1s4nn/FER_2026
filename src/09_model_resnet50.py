# -*- coding: utf-8 -*-
"""
09_model_resnet50.py

Here I switch to transfer learning. I start from ResNet50 weights
pretrained on ImageNet and only unfreeze the last conv block (layer4)
plus the final classifier head. I keep the earlier layers frozen on
purpose, the low-level edge and texture filters ResNet already learned
from ImageNet are useful for faces too, and fine-tuning the whole network
on a dataset as small as CK+ would just overfit fast.

I replace the original 1000-class ImageNet head with a small two-layer
head sized for however many emotion classes are in the dataset, with
dropout in between to keep it from memorizing the training faces.

Run this after 02_preprocess_data.py:
    python src/09_model_resnet50.py
"""

import os
import sys

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANIFEST_CSV, MODELS_DIR, FIGURES_DIR, EMOTION_LABELS, ensure_dirs
from utils import (set_seed, get_device, FERDataset, get_train_transform,
                    get_eval_transform, train_deep_model, evaluate_deep_model,
                    plot_training_curves, plot_confusion_matrix, save_metrics_row)

MODEL_NAME = 'ResNet50'
EPOCHS = 30
LR = 1e-4   # lower than the CNN's LR since I'm fine-tuning pretrained weights
BATCH_SIZE = 32
METRICS_CSV = os.path.join(os.path.dirname(FIGURES_DIR), 'metrics_summary.csv')


def build_model(num_classes):
    model = models.resnet50(weights='IMAGENET1K_V1')

    # I only let layer4 and the final fc head update, everything before
    # that stays frozen at its ImageNet weights
    for name, param in model.named_parameters():
        param.requires_grad = any(x in name for x in ['layer4', 'fc'])

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


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

    model = build_model(num_classes=len(EMOTION_LABELS))
    model, history = train_deep_model(
        model, MODEL_NAME, train_loader, val_loader, device, MODELS_DIR,
        epochs=EPOCHS, lr=LR
    )

    acc, preds, trues = evaluate_deep_model(model, MODEL_NAME, test_loader, device, EMOTION_LABELS)

    plot_training_curves(
        history, MODEL_NAME, '#e74c3c',
        os.path.join(FIGURES_DIR, 'resnet50_training_curves.png')
    )
    plot_confusion_matrix(
        trues, preds, EMOTION_LABELS, MODEL_NAME, acc,
        os.path.join(FIGURES_DIR, 'resnet50_confusion_matrix.png')
    )
    save_metrics_row(METRICS_CSV, MODEL_NAME, 'Deep Learning', trues, preds)


if __name__ == '__main__':
    main()
