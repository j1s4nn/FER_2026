# -*- coding: utf-8 -*-
"""
train_all.py

The one-command runner for the whole project. Instead of opening seven
terminals and remembering the right order myself, this script runs every
step of the pipeline in sequence: feature engineering for the classical
baselines, then all seven models (KNN, SVM, custom CNN, VGG16,
MobileNetV2, ResNet50, EfficientNetB0), and finally the comparison
figures.

I run each step in its own subprocess on purpose. Deep models hold onto
GPU memory until their process exits, so if I trained all five backbones
inside one long Python process the second or third model would start
fighting the first one for VRAM. Fresh process per model keeps every run
clean, and if one model crashes the ones before it are already saved.

It expects the dataset to be ready (01_download_data.py and
02_preprocess_data.py). If manifest.csv is missing it will try to run
those two first, but the download step needs a Kaggle token, see the
docstring in src/01_download_data.py for how to set that up.

Run it from the project root:
    python train_all.py
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, 'src')
MANIFEST = os.path.join(BASE_DIR, 'data', 'processed', 'manifest.csv')

# Every model trains in its own process, in this order. Classical baselines
# first (fast), then the deep models from smallest to biggest so I see
# results appearing early while the heavy backbones are still running.
PIPELINE = [
    ('Feature engineering (flat features for KNN/SVM)', '03_feature_engineering.py'),
    ('KNN baseline',                                    '04_model_knn.py'),
    ('SVM baseline',                                    '05_model_svm.py'),
    ('Custom CNN (from scratch)',                       '06_model_cnn.py'),
    ('VGG16 (transfer learning)',                       '07_model_vgg16.py'),
    ('MobileNetV2 (transfer learning)',                 '08_model_mobilenetv2.py'),
    ('ResNet50 (transfer learning)',                    '09_model_resnet50.py'),
    ('EfficientNetB0 (transfer learning)',              '10_model_efficientnet.py'),
    ('Comparison figures',                              '11_compare_models.py'),
]


def run_step(script):
    """Run one pipeline script with the same Python interpreter I was
    started with, streaming its output live into my console."""
    path = os.path.join(SRC, script)
    print('\n' + '=' * 72)
    print(f'RUNNING  {script}')
    print('=' * 72)
    result = subprocess.run([sys.executable, path], cwd=BASE_DIR)
    return result.returncode == 0


def main():
    os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)

    if not os.path.exists(MANIFEST):
        print('manifest.csv not found, running download + preprocessing first...')
        for script in ('01_download_data.py', '02_preprocess_data.py'):
            if not run_step(script):
                print(f'Stopping: {script} failed. Fix the error above and re-run.')
                return
        if not os.path.exists(MANIFEST):
            print('Still no manifest.csv after preprocessing, stopping.')
            return

    failed = []
    for title, script in PIPELINE:
        print(f'\n>>> Next step: {title}')
        if not run_step(script):
            failed.append(script)
            print(f'!!! {script} failed. Continuing with the remaining steps,')
            print('    check the error output above.')

    print('\n' + '=' * 72)
    if failed:
        print(f'train_all.py finished with {len(failed)} failed step(s):')
        for s in failed:
            print(f'  - {s}')
        print('Re-run those scripts individually to see the full error.')
    else:
        print('train_all.py finished successfully.')
        print('All metrics are in metrics_summary.csv and every figure is')
        print('in figures/. Run src/12_live_demo.py to try the models live.')
    print('=' * 72)


if __name__ == '__main__':
    main()
