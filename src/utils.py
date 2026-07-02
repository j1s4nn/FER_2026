# -*- coding: utf-8 -*-
"""
utils.py

I put all the small helper pieces I reuse in more than one script here,
things like the PyTorch dataset class, the transforms, and the plotting
helpers for training curves. Keeping them in one shared file means every
model script builds its charts the same way, so the final comparison
figures actually line up with each other.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED, IMG_SIZE_DL


def set_seed(seed=SEED):
    """I call this at the start of every training script so my results
    are repeatable between runs, otherwise I can never tell if a change
    in accuracy came from my code or just random initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


# ── Publication-style plotting defaults ───────────────────────────────────
def apply_plot_style():
    """I like all my figures to share the same look, so I set these once
    instead of repeating the same rcParams block in five different files."""
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# ── Dataset & transforms shared by every deep model ────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size=IMG_SIZE_DL):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def get_eval_transform(img_size=IMG_SIZE_DL):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


class FERDataset(Dataset):
    """A plain image-path dataset. I load the image lazily in __getitem__
    instead of loading everything into memory up front, since the
    augmented set can get into the thousands of files."""

    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ── Training loop shared by CNN / ResNet50 / EfficientNetB0 ────────────────
def train_deep_model(model, name, train_loader, val_loader, device,
                      models_dir, epochs=30, lr=1e-3):
    """One training loop that every deep model script calls. I found early
    on that copy-pasting this into each file just meant fixing the same bug
    three times, so now it lives here."""
    import torch.optim as optim

    model = model.to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_path = os.path.join(models_dir, f'{name}_best.pth')

    for epoch in range(epochs):
        # training pass
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_correct += out.max(1)[1].eq(lbls).sum().item()
            tr_total += lbls.size(0)

        # validation pass, no gradient tracking needed here
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                vl_loss += criterion(out, lbls).item()
                vl_correct += out.max(1)[1].eq(lbls).sum().item()
                vl_total += lbls.size(0)

        t_acc = 100 * tr_correct / tr_total
        v_acc = 100 * vl_correct / vl_total
        t_l = tr_loss / len(train_loader)
        v_l = vl_loss / len(val_loader)

        history['train_loss'].append(t_l)
        history['val_loss'].append(v_l)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        # I only keep the checkpoint that scored best on validation, so a
        # bad late epoch never overwrites a good earlier one
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'[{name}] epoch {epoch+1:02d}/{epochs}  '
                  f'train_loss={t_l:.4f}  val_loss={v_l:.4f}  '
                  f'train_acc={t_acc:.2f}%  val_acc={v_acc:.2f}%  '
                  f'best={best_val_acc:.2f}%')

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, history


def evaluate_deep_model(model, name, test_loader, device, class_names):
    from sklearn.metrics import accuracy_score, classification_report

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            p = model(imgs.to(device)).max(1)[1].cpu().numpy()
            preds.extend(p)
            trues.extend(lbls.numpy())
    preds, trues = np.array(preds), np.array(trues)
    acc = accuracy_score(trues, preds) * 100
    print(f'\n{name} test accuracy: {acc:.2f}%')
    print(classification_report(trues, preds, target_names=class_names, digits=4))
    return acc, preds, trues


def plot_training_curves(history, model_name, color, save_path):
    """I draw the loss and accuracy curves side by side, this is the same
    two-panel layout I used for every deep model so they're easy to
    compare visually when I put them side by side later."""
    apply_plot_style()
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{model_name} — Training History', fontsize=14, fontweight='bold')

    axes[0].plot(epochs_range, history['train_loss'], color=color, lw=2, label='Train Loss')
    axes[0].plot(epochs_range, history['val_loss'], color=color, lw=2,
                 linestyle='--', marker='o', markersize=3, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(epochs_range, history['train_acc'], color=color, lw=2, label='Train Acc')
    axes[1].plot(epochs_range, history['val_acc'], color=color, lw=2,
                 linestyle='--', marker='o', markersize=3, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 105)
    axes[1].set_title('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved training curve to {save_path}')


def plot_confusion_matrix(trues, preds, class_names, model_name, acc, save_path):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    apply_plot_style()
    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, annot_kws={'size': 10})
    ax.set_title(f'{model_name} — Confusion Matrix (Test Acc: {acc:.1f}%)',
                 fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved confusion matrix to {save_path}')


def save_metrics_row(csv_path, model_name, model_type, trues, preds):
    """I append one row of accuracy / precision / recall / F1 to a shared
    CSV so compare_models.py doesn't need to re-run every model, it just
    reads whatever rows have piled up here."""
    import pandas as pd
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)

    row = {
        'Model': model_name,
        'Type': model_type,
        'Accuracy': accuracy_score(trues, preds) * 100,
        'Precision': precision_score(trues, preds, average='macro') * 100,
        'Recall': recall_score(trues, preds, average='macro') * 100,
        'F1-Score': f1_score(trues, preds, average='macro') * 100,
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['Model'] != model_name]   # replace a stale row if I re-ran a model
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    print(f'Recorded metrics for {model_name} in {csv_path}')
    return row
