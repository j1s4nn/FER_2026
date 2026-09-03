# -*- coding: utf-8 -*-
"""
07_model_vgg16.py

My third deep model and the heavyweight of the comparison: VGG16. It is a
plain stack of 3x3 convolutions, thirteen of them, no skip connections and
no fancy blocks, which makes it a nice contrast to ResNet50 and
EfficientNetB0 in the study. The question I want this model to answer is
whether a deep-but-simple architecture can keep up with the modern
block-based designs on a dataset as small as CK+.

Same fine-tuning strategy as the other transfer-learning scripts: I start
from ImageNet weights, keep the early convolutional layers frozen, and only
unfreeze the last conv block (features.24 / 26 / 28, the three conv-relu
pairs of block five) plus the classifier head. The original 1000-class
ImageNet head gets replaced with a two-layer head sized for the emotion
classes, with dropout in between.

VGG16 has roughly 138M parameters, so it is the most expensive model in the
project to train, but on a 12GB GPU at 224x224 it still fits comfortably
at batch size 32.

Run this after 02_preprocess_data.py:
    python src/07_model_vgg16.py
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

MODEL_NAME = 'VGG16'
EPOCHS = 30
LR = 1e-4   # same low LR as the other pretrained backbones
BATCH_SIZE = 32
METRICS_CSV = os.path.join(os.path.dirname(FIGURES_DIR), 'metrics_summary.csv')


def build_model(num_classes):
    model = models.vgg16(weights='IMAGENET1K_V1')

    # I only let the last conv block (features.24/26/28) and the classifier
    # head update, everything before that stays at its ImageNet weights
    for name, param in model.named_parameters():
        param.requires_grad = any(
            x in name for x in ['features.24', 'features.26', 'features.28', 'classifier']
        )

    # classifier[6] is the original Linear(4096, 1000) layer, I swap it for
    # a smaller two-layer head sized for this dataset
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
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
        history, MODEL_NAME, '#9b59b6',
        os.path.join(FIGURES_DIR, 'vgg16_training_curves.png')
    )
    plot_confusion_matrix(
        trues, preds, EMOTION_LABELS, MODEL_NAME, acc,
        os.path.join(FIGURES_DIR, 'vgg16_confusion_matrix.png')
    )
    save_metrics_row(METRICS_CSV, MODEL_NAME, 'Deep Learning', trues, preds)


if __name__ == '__main__':
    main()
