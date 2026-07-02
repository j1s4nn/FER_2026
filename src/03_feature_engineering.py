# -*- coding: utf-8 -*-
"""
03_feature_engineering.py

KNN and SVM don't work directly on raw image folders the way a CNN does,
they need a flat numeric vector per image. So in this step I take every
image listed in manifest.csv, convert it to grayscale, resize it down to
a small fixed size, flatten it into a 1D vector, and then standardize
everything with a scaler fit only on the training split (never on val or
test, that would leak information from the split I'm supposed to be
evaluating on).

I save the resulting arrays to disk so 04_model_knn.py and 05_model_svm.py
don't have to redo this every time I re-run them.

Run this after 02_preprocess_data.py:
    python src/03_feature_engineering.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANIFEST_CSV, FLAT_FEATURES_DIR, IMG_SIZE_FLAT, ensure_dirs


def load_flat(paths, size=IMG_SIZE_FLAT):
    """Grayscale + resize + flatten. This throws away color and a lot of
    spatial structure, which is exactly why KNN/SVM top out lower than the
    CNN-based models later, but it's a fair classical-ML baseline."""
    X = []
    for p in paths:
        img = Image.open(p).convert('L').resize((size, size))
        X.append(np.array(img, dtype=np.float32).flatten())
    return np.array(X)


def main():
    ensure_dirs()

    if not os.path.exists(MANIFEST_CSV):
        print('manifest.csv not found, run 02_preprocess_data.py first.')
        return

    df = pd.read_csv(MANIFEST_CSV)
    print(f'Loaded manifest with {len(df)} rows.')

    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']
    test_df  = df[df['split'] == 'test']

    print('Building flattened features (this reads every image once)...')
    X_train = load_flat(train_df['path'].tolist())
    X_val   = load_flat(val_df['path'].tolist())
    X_test  = load_flat(test_df['path'].tolist())

    y_train = train_df['label_idx'].to_numpy()
    y_val   = val_df['label_idx'].to_numpy()
    y_test  = test_df['label_idx'].to_numpy()

    # I fit the scaler on training data only, then apply the same
    # transform to val/test so there's no leakage from the splits I use
    # for model selection and final evaluation.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    print(f'Feature shape: {X_train_sc.shape}')

    np.save(os.path.join(FLAT_FEATURES_DIR, 'X_train.npy'), X_train_sc)
    np.save(os.path.join(FLAT_FEATURES_DIR, 'X_val.npy'), X_val_sc)
    np.save(os.path.join(FLAT_FEATURES_DIR, 'X_test.npy'), X_test_sc)
    np.save(os.path.join(FLAT_FEATURES_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(FLAT_FEATURES_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(FLAT_FEATURES_DIR, 'y_test.npy'), y_test)

    with open(os.path.join(FLAT_FEATURES_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f'Saved flattened feature arrays and scaler to {FLAT_FEATURES_DIR}')


if __name__ == '__main__':
    main()
