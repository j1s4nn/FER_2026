# -*- coding: utf-8 -*-
"""
04_model_knn.py

My first baseline model. KNN doesn't really "learn" a decision boundary,
it just memorizes the training features and votes among the nearest
neighbors at prediction time. I use it here mostly as a sanity check, if
my fancier deep models can't beat this, something is wrong with them.

I settled on k=3 after trying a handful of values (3, 5, 7, 9) on the
validation split, 3 gave the best balance without obviously overfitting
to noisy neighbors.

Run this after 03_feature_engineering.py:
    python src/04_model_knn.py
"""

import os
import sys
import pickle

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FLAT_FEATURES_DIR, MODELS_DIR, FIGURES_DIR, EMOTION_LABELS, ensure_dirs
from utils import plot_confusion_matrix, save_metrics_row

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

MODEL_NAME = 'KNN'
K_NEIGHBORS = 3
METRICS_CSV = os.path.join(os.path.dirname(FIGURES_DIR), 'metrics_summary.csv')


def main():
    ensure_dirs()

    X_train = np.load(os.path.join(FLAT_FEATURES_DIR, 'X_train.npy'))
    X_test  = np.load(os.path.join(FLAT_FEATURES_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(FLAT_FEATURES_DIR, 'y_train.npy'))
    y_test  = np.load(os.path.join(FLAT_FEATURES_DIR, 'y_test.npy'))

    print(f'Training KNN with k={K_NEIGHBORS} on {X_train.shape[0]} samples...')
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric='euclidean', n_jobs=-1)
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f'\nKNN test accuracy: {acc:.2f}%')
    print(classification_report(y_test, preds, target_names=EMOTION_LABELS))

    # I save the fitted model with pickle since scikit-learn models don't
    # use the .pth checkpoint format the PyTorch models do.
    model_path = os.path.join(MODELS_DIR, 'knn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    print(f'Saved model to {model_path}')

    plot_confusion_matrix(
        y_test, preds, EMOTION_LABELS, MODEL_NAME, acc,
        os.path.join(FIGURES_DIR, 'knn_confusion_matrix.png')
    )

    save_metrics_row(METRICS_CSV, MODEL_NAME, 'Classical ML', y_test, preds)


if __name__ == '__main__':
    main()
