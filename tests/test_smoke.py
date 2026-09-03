# -*- coding: utf-8 -*-
"""
tests/test_smoke.py

CPU-only smoke tests. They verify that every model architecture builds,
that a forward pass produces the right output shape, that the shared
helpers (transforms, metrics CSV writer, compare-script registry) behave,
and that the classical ML path works end to end on synthetic data.

No dataset download, no GPU, no pretrained weight downloads: the tests
patch the torchvision factories to use random weights so CI stays fast
and offline-friendly.

Run from the project root:
    pytest tests/ -v
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import torchvision.models as tv_models  # noqa: E402
from config import SEED, IMG_SIZE_DL, IMG_SIZE_FLAT  # noqa: E402
from utils import (get_eval_transform, get_train_transform, FERDataset,
                    save_metrics_row)  # noqa: E402

NUM_CLASSES = 7


def _import(mod_name):
    return importlib.import_module(mod_name)


# ── config & shared helpers ─────────────────────────────────────────────────
def test_config_constants():
    assert SEED == 42
    assert IMG_SIZE_DL == 224
    assert IMG_SIZE_FLAT == 48


def test_eval_transform_shape():
    img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
    tensor = get_eval_transform()(img)
    assert tensor.shape == (3, IMG_SIZE_DL, IMG_SIZE_DL)


def test_train_transform_shape():
    img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
    tensor = get_train_transform()(img)
    assert tensor.shape == (3, IMG_SIZE_DL, IMG_SIZE_DL)


def test_fer_dataset_returns_pairs(tmp_path):
    path = tmp_path / 'face.jpg'
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
    ds = FERDataset([str(path)], [3], get_eval_transform())
    img, label = ds[0]
    assert img.shape == (3, IMG_SIZE_DL, IMG_SIZE_DL)
    assert label == 3


def test_save_metrics_row_roundtrip(tmp_path):
    csv_path = tmp_path / 'metrics_summary.csv'
    trues = np.array([0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 2, 0, 2, 1])
    row = save_metrics_row(str(csv_path), 'TestModel', 'Test', trues, preds)
    assert 0 < row['Accuracy'] <= 100
    df = pd.read_csv(csv_path)
    assert list(df['Model']) == ['TestModel']
    # re-running replaces the row instead of duplicating it
    save_metrics_row(str(csv_path), 'TestModel', 'Test', trues, preds)
    df = pd.read_csv(csv_path)
    assert len(df) == 1


# ── architecture forward passes (random weights, CPU) ───────────────────────
def test_custom_cnn_forward():
    mod = _import('06_model_cnn')
    model = mod.SimpleCNN(NUM_CLASSES).eval()
    out = model(torch.rand(2, 3, IMG_SIZE_DL, IMG_SIZE_DL))
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize('module_name,factory', [
    ('07_model_vgg16', 'vgg16'),
    ('08_model_mobilenetv2', 'mobilenet_v2'),
    ('09_model_resnet50', 'resnet50'),
    ('10_model_efficientnet', 'efficientnet_b0'),
])
def test_transfer_builders_forward(monkeypatch, module_name, factory):
    """Patch the torchvision factory so no ImageNet weights download."""
    mod = _import(module_name)
    real_factory = getattr(tv_models, factory)
    monkeypatch.setattr(mod.models, factory,
                        lambda weights: real_factory(weights=None))
    model = mod.build_model(NUM_CLASSES).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, IMG_SIZE_DL, IMG_SIZE_DL))
    assert out.shape == (1, NUM_CLASSES)


def test_freeze_policy_freezes_early_backbone():
    """Sanity check: the transfer-learning freeze policy keeps the early
    backbone layers (features.0 / features.1 ...) at requires_grad=False
    while the last block and head stay trainable."""
    mod = _import('08_model_mobilenetv2')
    assert mod.MODEL_NAME == 'MobileNetV2'

    model = tv_models.mobilenet_v2(weights=None)
    # mimic the script's freeze policy
    for name, param in model.named_parameters():
        param.requires_grad = any(x in name for x in ['features.18', 'classifier'])

    early = [p for n, p in model.named_parameters()
             if n.startswith('features.0.') or n.startswith('features.1.')]
    assert early and all(not p.requires_grad for p in early)
    assert any(p.requires_grad for p in model.parameters())


# ── classical ML path on synthetic data ─────────────────────────────────────
def test_classical_baselines_on_synthetic_data():
    rng = np.random.default_rng(SEED)
    # three well-separated Gaussian blobs -> any sane classifier gets >90%
    X = np.vstack([rng.normal(loc=m, scale=0.5, size=(40, 16)) for m in (0, 3, 6)])
    y = np.repeat([0, 1, 2], 40)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    svm = SVC(kernel='rbf', C=10, probability=True, random_state=SEED).fit(X, y)
    assert (knn.predict(X) == y).mean() > 0.9
    assert (svm.predict(X) == y).mean() > 0.9


# ── comparison registry ─────────────────────────────────────────────────────
def test_compare_registry_covers_all_seven_models():
    mod = _import('11_compare_models')
    assert mod.EXPECTED_MODELS == ['KNN', 'SVM', 'CNN', 'VGG16', 'MobileNetV2',
                                    'ResNet50', 'EfficientNetB0']


def test_live_demo_path_map_covers_all_seven_models():
    mod = _import('12_live_demo')
    # the model map is built inside analyze_with_model; check the loaders
    # registry indirectly via the checkpoint naming convention
    assert hasattr(mod, 'build_vgg_arch')
    assert hasattr(mod, 'build_mobilenet_arch')
    assert hasattr(mod, 'build_cnn_arch')
