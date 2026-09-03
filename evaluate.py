# -*- coding: utf-8 -*-
"""
evaluate.py

The no-retraining report generator. Once the models have been trained at
least once (their checkpoints live in models/), this script rebuilds the
entire results story from disk:

  1. loads the saved KNN / SVM pickles and the five deep checkpoints
  2. re-runs each of them on the untouched test split
  3. rewrites metrics_summary.csv from scratch (so a stale row from an
     old run can never survive)
  4. regenerates every confusion matrix in figures/
  5. regenerates the comparison charts via src/11_compare_models.py

I use this whenever I want fresh comparison figures after, say, swapping
in a new test split, without paying for hours of GPU training again.

Run it from the project root, after the models have been trained:
    python evaluate.py
"""

import importlib
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import (MANIFEST_CSV, MODELS_DIR, FIGURES_DIR, FLAT_FEATURES_DIR,
                     EMOTION_LABELS, ensure_dirs)
from utils import (get_device, FERDataset, get_eval_transform,
                    evaluate_deep_model, plot_confusion_matrix, save_metrics_row)

METRICS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'metrics_summary.csv')

# (display name, src module that defines the architecture, checkpoint file,
#  how to build the model from that module)
def _cnn_builder(mod, n):
    return mod.SimpleCNN(n)


def _generic_builder(mod, n):
    return mod.build_model(n)


DL_MODELS = [
    ('CNN',            '06_model_cnn',        'CNN_best.pth',            _cnn_builder),
    ('VGG16',          '07_model_vgg16',      'VGG16_best.pth',          _generic_builder),
    ('MobileNetV2',    '08_model_mobilenetv2','MobileNetV2_best.pth',    _generic_builder),
    ('ResNet50',       '09_model_resnet50',   'ResNet50_best.pth',       _generic_builder),
    ('EfficientNetB0', '10_model_efficientnet','EfficientNetB0_best.pth', _generic_builder),
]

CM_FILES = {
    'KNN': 'knn_confusion_matrix.png',
    'SVM': 'svm_confusion_matrix.png',
    'CNN': 'cnn_confusion_matrix.png',
    'VGG16': 'vgg16_confusion_matrix.png',
    'MobileNetV2': 'mobilenetv2_confusion_matrix.png',
    'ResNet50': 'resnet50_confusion_matrix.png',
    'EfficientNetB0': 'efficientnet_confusion_matrix.png',
}


def evaluate_classical(device=None):
    """KNN and SVM are cheap to re-score: load the pickle, predict on the
    saved test features, done. No fitting happens here."""
    X_test = np.load(os.path.join(FLAT_FEATURES_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(FLAT_FEATURES_DIR, 'y_test.npy'))

    for name, filename in (('KNN', 'knn_model.pkl'), ('SVM', 'svm_model.pkl')):
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f'[skip] {name}: no saved model at {path}')
            continue
        with open(path, 'rb') as f:
            model = pickle.load(f)
        preds = model.predict(X_test)
        acc = (preds == y_test).mean() * 100
        print(f'{name} test accuracy: {acc:.2f}%')
        plot_confusion_matrix(
            y_test, preds, EMOTION_LABELS, name, acc,
            os.path.join(FIGURES_DIR, CM_FILES[name])
        )
        save_metrics_row(METRICS_CSV, name, 'Classical ML', y_test, preds)


def evaluate_deep_models(device):
    """Rebuild each architecture (random weights), overwrite them with the
    saved best checkpoints, and score them on the test loader."""
    if not os.path.exists(MANIFEST_CSV):
        print('[skip] deep models: manifest.csv not found.')
        return

    df = pd.read_csv(MANIFEST_CSV)
    test_df = df[df['split'] == 'test']
    test_ds = FERDataset(test_df['path'].tolist(),
                          test_df['label_idx'].tolist(), get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    for name, module_name, ckpt, builder in DL_MODELS:
        ckpt_path = os.path.join(MODELS_DIR, ckpt)
        if not os.path.exists(ckpt_path):
            print(f'[skip] {name}: no checkpoint at {ckpt_path}')
            continue

        mod = importlib.import_module(module_name)
        model = builder(mod, len(EMOTION_LABELS))
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)

        acc, preds, trues = evaluate_deep_model(
            model, name, test_loader, device, EMOTION_LABELS)
        plot_confusion_matrix(
            trues, preds, EMOTION_LABELS, name, acc,
            os.path.join(FIGURES_DIR, CM_FILES[name])
        )
        save_metrics_row(METRICS_CSV, name, 'Deep Learning', trues, preds)

        # free the GPU before the next backbone loads
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def regenerate_comparison():
    """11_compare_models.py reads metrics_summary.csv and redraws the
    bar charts, I just call its main() like the standalone script would."""
    mod = importlib.import_module('11_compare_models')
    mod.main()


def main():
    ensure_dirs()
    device = get_device()

    # rebuild the CSV from scratch so no stale row survives
    if os.path.exists(METRICS_CSV):
        os.remove(METRICS_CSV)

    evaluate_classical(device)
    evaluate_deep_models(device)
    regenerate_comparison()

    print('\nevaluate.py complete. metrics_summary.csv and figures/ are up to date.')


if __name__ == '__main__':
    main()
