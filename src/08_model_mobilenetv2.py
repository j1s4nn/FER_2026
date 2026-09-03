# -*- coding: utf-8 -*-
"""
08_model_mobilenetv2.py

My fourth deep model and the lightweight champion of the comparison:
MobileNetV2. It is built from inverted residual blocks with depthwise
separable convolutions, designed for phones and embedded devices, so it
only has about 3.5M parameters. I include it because a fair model
comparison is not only about who reaches the highest accuracy, it is also
about how much model you had to pay for that accuracy. If MobileNetV2 gets
close to ResNet50 with a fraction of the parameters, that is a genuinely
useful result for anyone who wants to deploy FER on a real device.

Same fine-tuning strategy as the EfficientNetB0 script: I unfreeze only
the last feature block (features.18, the final 1x1 conv expansion layer)
plus the classifier head, and leave everything else at its ImageNet
weights.

Run this after 02_preprocess_data.py:
    python src/08_model_mobilenetv2.py
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

MODEL_NAME = 'MobileNetV2'
EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 32
METRICS_CSV = os.path.join(os.path.dirname(FIGURES_DIR), 'metrics_summary.csv')


def build_model(num_classes):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # features.18 is the last conv block of the network, I let it and the
    # classifier head train while the rest stays frozen
    for name, param in model.named_parameters():
        param.requires_grad = any(
            x in name for x in ['features.18', 'classifier']
        )

    # classifier is Sequential(Dropout, Linear(1280, 1000)), I only resize
    # the final projection layer to the number of emotion classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
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
        history, MODEL_NAME, '#f39c12',
        os.path.join(FIGURES_DIR, 'mobilenetv2_training_curves.png')
    )
    plot_confusion_matrix(
        trues, preds, EMOTION_LABELS, MODEL_NAME, acc,
        os.path.join(FIGURES_DIR, 'mobilenetv2_confusion_matrix.png')
    )
    save_metrics_row(METRICS_CSV, MODEL_NAME, 'Deep Learning', trues, preds)


if __name__ == '__main__':
    main()
