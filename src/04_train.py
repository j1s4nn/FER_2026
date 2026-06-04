"""
04_train.py
===========
Course  : Digital Image Processing
Major   : Artificial Intelligence
Name    : Hossen Md Jisan
ID      : 202353460019
Topic   : Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

Description
-----------
I train PoleZeroCNN, FreqMLP, and FusionNet on the generated dataset.
For each model I:
  - Split data into 70% train / 15% val / 15% test
  - Train with Adam optimizer + cosine annealing LR schedule
  - Save the best checkpoint (by validation accuracy)
  - Log and plot training/validation loss and accuracy curves

All figures and checkpoints are saved automatically.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUT_FIG_DIR = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# ── Dynamic import of 03_model.py ────────────────────────────────────────────
import importlib.util

def _import_model():
    spec = importlib.util.spec_from_file_location(
        "model_module",
        os.path.join(BASE_DIR, "src", "03_model.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod = _import_model()
build_model = _mod.build_model


# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (remainder)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class PoleZeroDataset(Dataset):
    def __init__(self, images, freq_resp, labels):
        # images: (N, H, W) float32  → add channel dim
        self.images    = torch.tensor(images[:, None, :, :], dtype=torch.float32)
        self.freq_resp = torch.tensor(freq_resp, dtype=torch.float32)
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.freq_resp[idx], self.labels[idx]


def load_data():
    path = os.path.join(DATASET_DIR, "dataset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset not found. Run 01_dataset_generation.py first."
        )
    d = np.load(path)
    return d["images"], d["freq_resp"], d["labels"]


def get_dataloaders():
    images, freq_resp, labels = load_data()
    dataset = PoleZeroDataset(images, freq_resp, labels)
    n = len(dataset)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    return (
        make_loader(train_ds, True),
        make_loader(val_ds,   False),
        make_loader(test_ds,  False),
    )


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, model_name):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, freq, labels in loader:
        images = images.to(DEVICE)
        freq   = freq.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        if model_name == "fusion":
            logits = model(images, freq)
        elif model_name == "cnn":
            logits = model(images)
        else:
            logits = model(freq)

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, model_name):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, freq, labels in loader:
        images = images.to(DEVICE)
        freq   = freq.to(DEVICE)
        labels = labels.to(DEVICE)

        if model_name == "fusion":
            logits = model(images, freq)
        elif model_name == "cnn":
            logits = model(images)
        else:
            logits = model(freq)

        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


# ── Plot training curves ──────────────────────────────────────────────────────
def plot_curves(history, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=1.5, linestyle="--")
    axes[0].set_title(f"{model_name.upper()} — Loss", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   linewidth=1.5, linestyle="--")
    axes[1].set_title(f"{model_name.upper()} — Accuracy", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_FIG_DIR, f"fig_train_curves_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Main training function ────────────────────────────────────────────────────
def train_model(model_name, train_loader, val_loader):
    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*55}")

    model = build_model(model_name).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}  |  Device: {DEVICE}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path = os.path.join(DATASET_DIR, f"best_{model_name}.pt")

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, model_name)
        v_loss, v_acc = evaluate(model, val_loader, criterion, model_name)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Ep {epoch:3d}/{EPOCHS} | "
                f"Loss {t_loss:.4f}/{v_loss:.4f} | "
                f"Acc {t_acc:.3f}/{v_acc:.3f} | "
                f"LR {lr_now:.2e}"
            )

    print(f"  Best Val Acc: {best_val_acc:.4f}")
    plot_curves(history, model_name)

    # Save history
    hist_path = os.path.join(DATASET_DIR, f"history_{model_name}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return ckpt_path, best_val_acc


def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    results = {}
    for name in ["cnn", "mlp", "fusion"]:
        ckpt, best_acc = train_model(name, train_loader, val_loader)
        results[name] = {"checkpoint": ckpt, "best_val_acc": best_acc}

    print("\n[SUMMARY] Best Validation Accuracies:")
    for name, info in results.items():
        print(f"  {name.upper():8s}: {info['best_val_acc']:.4f}")

    summary_path = os.path.join(DATASET_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {summary_path}")
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
