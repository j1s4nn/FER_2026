# -*- coding: utf-8 -*-
"""
config.py

I keep every path and constant that the other scripts need in one place.
I did it this way so if I ever move the dataset or rename a folder, I only
have to change it here instead of hunting through eight different files.
"""

import os

# ── Base paths ────────────────────────────────────────────────────────────
# I always resolve paths relative to this file so the project runs the same
# way no matter which folder I launch a script from.
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR   = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR  = os.path.join(DATA_DIR, 'processed')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
FIGURES_DIR    = os.path.join(BASE_DIR, 'figures')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'output')

# I keep the raw Kaggle download and the split/augmented dataset separate
# on purpose, that way I can delete the processed folder and rebuild it
# without ever touching the original downloaded images.
CKPLUS_ROOT      = os.path.join(RAW_DATA_DIR, 'ckplus', 'CK+48')
AUGMENTED_DIR    = os.path.join(PROCESSED_DIR, 'augmented')
MANIFEST_CSV     = os.path.join(PROCESSED_DIR, 'manifest.csv')
FLAT_FEATURES_DIR = os.path.join(PROCESSED_DIR, 'flat_features')

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42

# ── Dataset settings ──────────────────────────────────────────────────────
# 350 images per class x 7 classes gives me a balanced 2,450 image set,
# this matched what worked well when I was experimenting in Colab.
TARGET_PER_CLASS = 350
IMG_SIZE_DL      = 224   # input size I feed into CNN / ResNet50 / EfficientNet
IMG_SIZE_FLAT    = 48    # smaller flattened size I use for KNN and SVM

# these get filled in automatically once I actually see the dataset folder,
# I don't hardcode the emotion names here because I want the code to adapt
# if I ever swap in a different dataset that has different class folders
EMOTION_LABELS = []
NUM_CLASSES = 0

def refresh_labels():
    """I call this after the raw dataset is confirmed to exist, so the
    label list always reflects what is actually on disk."""
    global EMOTION_LABELS, NUM_CLASSES
    if os.path.isdir(CKPLUS_ROOT):
        EMOTION_LABELS = sorted(os.listdir(CKPLUS_ROOT))
        NUM_CLASSES = len(EMOTION_LABELS)
    return EMOTION_LABELS, NUM_CLASSES

# I try to fill these in right away so other scripts can just "from config
# import EMOTION_LABELS" without extra ceremony, but it's fine if the
# dataset isn't downloaded yet, refresh_labels() can be re-run later.
refresh_labels()

# Colors I reuse across all the plots so every figure in the project looks
# consistent instead of matplotlib picking a different palette each time.
CLASS_COLORS = ['#e74c3c', '#9b59b6', '#3498db', '#1abc9c',
                 '#f39c12', '#e67e22', '#2ecc71']
MODEL_COLORS = {
    'KNN':             '#95a5a6',
    'SVM':             '#7f8c8d',
    'CNN':             '#2ecc71',
    'ResNet50':        '#e74c3c',
    'EfficientNetB0':  '#3498db',
}
ACCURACY_TARGET = 91  # the accuracy line I like to draw on my charts for reference

def ensure_dirs():
    """I run this at the top of most scripts so I never get a
    'no such file or directory' error just because a folder hasn't
    been created yet."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR,
              FIGURES_DIR, OUTPUT_DIR, AUGMENTED_DIR, FLAT_FEATURES_DIR]:
        os.makedirs(d, exist_ok=True)
