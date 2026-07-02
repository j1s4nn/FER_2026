# -*- coding: utf-8 -*-
"""
02_preprocess_data.py

Once the raw CK+ images are downloaded, I run this script to get them
into a shape my models can actually train on. Three things happen here:

  1. I balance the classes. CK+ is small and uneven across emotions, so
     I augment every class up to the same target count instead of letting
     the model learn a bias toward whichever emotion has the most photos.
  2. I split everything into train / val / test (80 / 10 / 10), stratified
     so each split keeps the same class balance as the full set.
  3. I write out a manifest.csv that records every image path, its label,
     and which split it belongs to, so every later script just reads this
     one file instead of re-scanning folders.

Run this after 01_download_data.py:
    python src/02_preprocess_data.py
"""

import os
import sys
import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CKPLUS_ROOT, AUGMENTED_DIR, MANIFEST_CSV, FIGURES_DIR,
                     TARGET_PER_CLASS, SEED, ensure_dirs, refresh_labels)
from utils import set_seed, apply_plot_style

import matplotlib.pyplot as plt

CLASS_COLORS = ['#e74c3c', '#9b59b6', '#3498db', '#1abc9c',
                 '#f39c12', '#e67e22', '#2ecc71']


def load_original_paths(emotion_labels):
    """I collect every image path and its numeric label from the raw
    CK+48 folder structure before I do anything else to it."""
    image_paths, labels = [], []
    for idx, emotion in enumerate(emotion_labels):
        emotion_dir = os.path.join(CKPLUS_ROOT, emotion)
        for f in sorted(os.listdir(emotion_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(emotion_dir, f))
                labels.append(idx)
    return image_paths, labels


def plot_class_distribution(counts, emotion_labels, save_path, title):
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    bars = axes[0].bar(emotion_labels, counts, color=CLASS_COLORS,
                        edgecolor='black', linewidth=0.7)
    axes[0].set_xlabel('Emotion Class')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Sample Count per Class')
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      str(c), ha='center', fontweight='bold', fontsize=10)

    axes[1].pie(counts, labels=emotion_labels, colors=CLASS_COLORS,
                autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
    axes[1].set_title('Class Proportion')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {save_path}')


def augment_to_balance(image_paths, labels, emotion_labels, target_per_class):
    """I top up every class to the same target count. Real images are
    copied over untouched first, then I generate augmented copies (flip,
    rotate, jitter, small affine shift) until each class reaches the
    target. This keeps the original photos intact and just fills the
    gap with realistic variations of them."""
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    aug_paths, aug_labels = [], []

    for class_idx, emotion in enumerate(emotion_labels):
        class_dir = os.path.join(AUGMENTED_DIR, emotion)
        os.makedirs(class_dir, exist_ok=True)
        originals = [(p, l) for p, l in zip(image_paths, labels) if l == class_idx]
        count = 0

        for p, _ in originals:
            img = Image.open(p).convert('RGB').resize((224, 224))
            save_path = os.path.join(class_dir, f'orig_{count}.jpg')
            img.save(save_path)
            aug_paths.append(save_path)
            aug_labels.append(class_idx)
            count += 1

        while count < target_per_class:
            src_path, _ = random.choice(originals)
            img = aug_transform(Image.open(src_path).convert('RGB'))
            save_path = os.path.join(class_dir, f'aug_{count}.jpg')
            img.save(save_path)
            aug_paths.append(save_path)
            aug_labels.append(class_idx)
            count += 1

        print(f'  {emotion}: {count} images (balanced)')

    return aug_paths, aug_labels


def build_manifest(aug_paths, aug_labels, emotion_labels):
    """I split the balanced set into train/val/test and record everything
    in one CSV. Downstream scripts filter this dataframe by the 'split'
    column instead of touching the filesystem directly."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        aug_paths, aug_labels, test_size=0.2, random_state=SEED, stratify=aug_labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    rows = []
    for paths, labs, split_name in [(X_train, y_train, 'train'),
                                     (X_val, y_val, 'val'),
                                     (X_test, y_test, 'test')]:
        for p, l in zip(paths, labs):
            rows.append({'path': p, 'label_idx': l,
                         'label_name': emotion_labels[l], 'split': split_name})

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_CSV, index=False)
    print(f'\nManifest written to {MANIFEST_CSV}')
    print(df['split'].value_counts())
    return df


def main():
    ensure_dirs()
    set_seed()

    emotion_labels, num_classes = refresh_labels()
    if num_classes == 0:
        print('I could not find the raw dataset. Run 01_download_data.py first.')
        return

    print('Classes:', emotion_labels)
    image_paths, labels = load_original_paths(emotion_labels)
    print(f'Original images found: {len(image_paths)}')

    original_counts = [Counter(labels)[i] for i in range(num_classes)]
    plot_class_distribution(
        original_counts, emotion_labels,
        os.path.join(FIGURES_DIR, 'fig1_class_distribution_original.png'),
        'CK+ Dataset — Original Class Distribution'
    )

    print(f'\nBalancing every class up to {TARGET_PER_CLASS} images...')
    aug_paths, aug_labels = augment_to_balance(
        image_paths, labels, emotion_labels, TARGET_PER_CLASS)

    aug_counts = [Counter(aug_labels)[i] for i in range(num_classes)]
    plot_class_distribution(
        aug_counts, emotion_labels,
        os.path.join(FIGURES_DIR, 'fig2_class_distribution_balanced.png'),
        f'CK+ Dataset After Balancing (Total: {sum(aug_counts)})'
    )

    build_manifest(aug_paths, aug_labels, emotion_labels)
    print('\nPreprocessing complete.')


if __name__ == '__main__':
    main()
