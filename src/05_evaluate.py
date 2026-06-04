"""
05_evaluate.py
==============
Course  : Digital Image Processing
Major   : Artificial Intelligence
Name    : Hossen Md Jisan
ID      : 202353460019
Topic   : Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

Description
-----------
I evaluate all trained models on the held-out test set and generate:
  - Classification report (precision, recall, F1, accuracy)
  - Confusion matrices (one per model)
  - PSNR & SSIM of reconstructed frequency responses vs. ground truth
  - Predicted vs. ground-truth pole-zero map comparison figures
  - Summary metrics table (CSV, JSON, and PNG bar chart)

All outputs are saved to output/figures/ and output/test_metrics/.
"""

import os
import sys
import json
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
FIG_DIR      = os.path.join(BASE_DIR, "output", "figures")
METRIC_DIR   = os.path.join(BASE_DIR, "output", "test_metrics")
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

CLASS_NAMES = ["lowpass", "highpass", "bandpass", "bandstop", "allpass"]
SEED        = 42
BATCH_SIZE  = 64
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PALETTE     = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]


# ── Dynamic imports ───────────────────────────────────────────────────────────
def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_model_mod = _import_module("model_module", os.path.join(BASE_DIR, "src", "03_model.py"))

build_model = _model_mod.build_model


# ── Inline PoleZeroDataset (avoids importing 04_train which has side-effects) ─
from torch.utils.data import Dataset

class PoleZeroDataset(Dataset):
    def __init__(self, images, freq_resp, labels):
        self.images    = torch.tensor(images[:, None, :, :], dtype=torch.float32)
        self.freq_resp = torch.tensor(freq_resp, dtype=torch.float32)
        self.labels    = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.freq_resp[idx], self.labels[idx]


# ── Data helpers ──────────────────────────────────────────────────────────────
def get_test_loader():
    path = os.path.join(DATASET_DIR, "dataset.npz")
    d    = np.load(path)
    dataset = PoleZeroDataset(d["images"], d["freq_resp"], d["labels"])
    n       = len(dataset)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    n_test  = n - n_train - n_val
    _, _, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def get_test_arrays():
    path = os.path.join(DATASET_DIR, "dataset.npz")
    d    = np.load(path)
    n       = len(d["labels"])
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    n_test  = n - n_train - n_val

    torch.manual_seed(SEED)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    test_idx = idx[n_train + n_val:].numpy()

    return (
        d["images"][test_idx],
        d["freq_resp"][test_idx],
        d["labels"][test_idx],
    )


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, model_name):
    model.eval()
    all_preds, all_labels = [], []
    for images, freq, labels in loader:
        images = images.to(DEVICE)
        freq   = freq.to(DEVICE)
        if model_name == "fusion":
            logits = model(images, freq)
        elif model_name == "cnn":
            logits = model(images)
        else:
            logits = model(freq)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── Figure: Confusion Matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(f"Confusion Matrix — {model_name.upper()}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"fig_confusion_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure: Predicted vs GT pole-zero maps ────────────────────────────────────
def plot_pz_predictions(images, y_true, y_pred, model_name, n_show=10):
    """
    Show side-by-side: pole-zero map image | GT label | Predicted label.
    """
    correct_mask   = y_true == y_pred
    incorrect_mask = ~correct_mask
    selected = []
    for m in [correct_mask, incorrect_mask]:
        idxs = np.where(m)[0][:n_show // 2]
        selected.extend(idxs)
    selected = selected[:n_show]

    n = len(selected)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=((n + 1) // 2 * 2.5, 6))
    axes = axes.flatten()
    fig.suptitle(f"Pole-Zero Map Predictions — {model_name.upper()}", fontsize=11, fontweight="bold")

    for i, idx in enumerate(selected):
        ax = axes[i]
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=1)
        gt   = CLASS_NAMES[y_true[idx]]
        pred = CLASS_NAMES[y_pred[idx]]
        color = "green" if gt == pred else "red"
        ax.set_title(f"GT: {gt}\nPred: {pred}", fontsize=7, color=color)
        ax.axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"fig_pz_predictions_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure: Per-class F1 bar chart ───────────────────────────────────────────
def plot_f1_per_class(y_true, y_pred, model_name):
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    f1_scores = [report[c]["f1-score"] for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, f1_scores, color=PALETTE, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-Score", fontsize=10)
    ax.set_title(f"Per-Class F1 Score — {model_name.upper()}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"fig_f1_per_class_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── PSNR / SSIM of freq response reconstruction ───────────────────────────────
def compute_psnr_ssim(freq_gt, y_true, y_pred):
    """
    Compare the mean frequency response of ground-truth class vs predicted class.
    Returns mean PSNR and SSIM as proxy reconstruction quality metrics.
    """
    psnr_list, ssim_list = [], []
    for cls_idx in range(len(CLASS_NAMES)):
        gt_mask   = (y_true == cls_idx)
        pred_mask = (y_pred == cls_idx)
        if gt_mask.sum() == 0 or pred_mask.sum() == 0:
            continue
        gt_mean   = freq_gt[gt_mask].mean(axis=0)
        pred_mean = freq_gt[pred_mask].mean(axis=0)
        # Normalize to [0, 1] for metric computation
        vmax = max(gt_mean.max(), pred_mean.max(), 1e-8)
        gt_n   = gt_mean   / vmax
        pred_n = pred_mean / vmax
        psnr_list.append(psnr_fn(gt_n, pred_n, data_range=1.0))
        ssim_list.append(ssim_fn(gt_n, pred_n, data_range=1.0))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# ── Figure: Model comparison bar chart ───────────────────────────────────────
def plot_model_comparison(summary_df):
    metrics  = ["accuracy", "precision", "recall", "f1", "psnr", "ssim"]
    n_models = len(summary_df)
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#264653", "#e9c46a", "#e76f51"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (_, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] for m in metrics]
        # Normalize PSNR to [0,1] range for display
        vals_disp = vals.copy()
        vals_disp[4] = vals[4] / 60.0  # scale PSNR/60 for display
        bars = ax.bar(x + i * width, vals_disp, width, label=row["model"].upper(),
                      color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "PSNR/60", "SSIM"], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison on Test Set", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(METRIC_DIR, "fig_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Main evaluation ───────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading test data ...")
    test_loader          = get_test_loader()
    images, freq, labels = get_test_arrays()

    summary_rows = []

    for model_name in ["cnn", "mlp", "fusion"]:
        ckpt_path = os.path.join(DATASET_DIR, f"best_{model_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] No checkpoint found for {model_name}. Run 04_train.py first.")
            continue

        print(f"\n[EVAL] {model_name.upper()} ...")
        model = build_model(model_name).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        y_pred, y_true = run_inference(model, test_loader, model_name)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        psnr_val, ssim_val = compute_psnr_ssim(freq, y_true, y_pred)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  PSNR     : {psnr_val:.2f} dB")
        print(f"  SSIM     : {ssim_val:.4f}")

        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
        report_path = os.path.join(METRIC_DIR, f"classification_report_{model_name}.txt")
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name.upper()}\n\n")
            f.write(report)
        print(f"[SAVED] {report_path}")

        plot_confusion_matrix(y_true, y_pred, model_name)
        plot_pz_predictions(images, y_true, y_pred, model_name)
        plot_f1_per_class(y_true, y_pred, model_name)

        summary_rows.append({
            "model"    : model_name,
            "accuracy" : round(acc,  4),
            "precision": round(prec, 4),
            "recall"   : round(rec,  4),
            "f1"       : round(f1,   4),
            "psnr"     : round(psnr_val, 2),
            "ssim"     : round(ssim_val, 4),
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path  = os.path.join(METRIC_DIR, "test_metrics_summary.csv")
        json_path = os.path.join(METRIC_DIR, "test_metrics_summary.json")
        df.to_csv(csv_path,   index=False)
        df.to_json(json_path, orient="records", indent=2)
        print(f"\n[SAVED] {csv_path}")
        print(f"[SAVED] {json_path}")
        print("\n[SUMMARY TABLE]")
        print(df.to_string(index=False))
        plot_model_comparison(df)

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
