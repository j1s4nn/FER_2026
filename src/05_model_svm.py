# -*- coding: utf-8 -*-
"""
05_model_svm.py

My second classical baseline. I use an RBF kernel here instead of a
linear one because facial expressions don't separate cleanly along
straight lines in raw pixel space, the RBF kernel lets the model draw
curved boundaries between the emotion classes.

C=10 came out of a small manual sweep (I tried 1, 10, 50, 100), C=10 gave
the best validation accuracy without the model just memorizing the
training set. probability=True costs a bit of training time but I need
the probability estimates later for the live demo's confidence scores.

Run this after 03_feature_engineering.py:
    python src/05_model_svm.py
"""

import os
import sys
import pickle

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FLAT_FEATURES_DIR, MODELS_DIR, FIGURES_DIR, EMOTION_LABELS, SEED, ensure_dirs
from utils import plot_confusion_matrix, save_metrics_row

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

MODEL_NAME = 'SVM'
METRICS_CSV = os.path.join(os.path.dirname(FIGURES_DIR), 'metrics_summary.csv')


def main():
    ensure_dirs()

    X_train = np.load(os.path.join(FLAT_FEATURES_DIR, 'X_train.npy'))
    X_test  = np.load(os.path.join(FLAT_FEATURES_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(FLAT_FEATURES_DIR, 'y_train.npy'))
    y_test  = np.load(os.path.join(FLAT_FEATURES_DIR, 'y_test.npy'))

    print(f'Training SVM (RBF kernel) on {X_train.shape[0]} samples...')
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED)
    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f'\nSVM test accuracy: {acc:.2f}%')
    print(classification_report(y_test, preds, target_names=EMOTION_LABELS))

    model_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)
    print(f'Saved model to {model_path}')

    plot_confusion_matrix(
        y_test, preds, EMOTION_LABELS, MODEL_NAME, acc,
        os.path.join(FIGURES_DIR, 'svm_confusion_matrix.png')
    )

    save_metrics_row(METRICS_CSV, MODEL_NAME, 'Classical ML', y_test, preds)


if __name__ == '__main__':
    main()
