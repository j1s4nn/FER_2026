"""
02_dataset_visualization.py
============================
Course  : Digital Image Processing
Major   : Artificial Intelligence
Name    : Hossen Md Jisan
ID      : 202353460019
Topic   : Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

Description
-----------
I visualize the generated dataset to confirm quality and gain intuition about
the distribution of pole-zero maps, frequency responses, and class balance.

All figures are saved automatically to output/dataset_viz/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from scipy.signal import freqz

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR    = os.path.join(BASE_DIR, "output", "dataset_viz")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["lowpass", "highpass", "bandpass", "bandstop", "allpass"]
PALETTE     = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]


def load_dataset():
    path = os.path.join(DATASET_DIR, "dataset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 01_dataset_generation.py first."
        )
    data = np.load(path)
    return (data["images"], data["freq_resp"],
            data["poles_vec"], data["zeros_vec"], data["labels"])


# ── Figure 1: Sample pole-zero maps per class ─────────────────────────────────
def plot_sample_pz_maps(images, labels, n_per_class=5):
    fig, axes = plt.subplots(
        N := len(CLASS_NAMES), n_per_class,
        figsize=(n_per_class * 2, N * 2),
    )
    fig.suptitle("Sample Pole-Zero Map Images by Class", fontsize=13, fontweight="bold")

    for row, (cls_idx, cls_name) in enumerate(enumerate(CLASS_NAMES)):
        idxs = np.where(labels == cls_idx)[0][:n_per_class]
        for col, idx in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(images[idx], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=9, labelpad=4)
        for col in range(len(idxs), n_per_class):
            axes[row, col].axis("off")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig01_sample_pz_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 2: Mean frequency response per class ───────────────────────────────
def plot_mean_freq_response(freq_resp, labels):
    fig, ax = plt.subplots(figsize=(8, 4))
    freqs = np.linspace(0, np.pi, freq_resp.shape[1])

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        mean_resp = freq_resp[mask].mean(axis=0)
        ax.plot(freqs, mean_resp, label=cls_name, color=PALETTE[cls_idx], linewidth=1.8)

    ax.set_xlabel("Normalized Frequency (rad/sample)", fontsize=10)
    ax.set_ylabel("Magnitude", fontsize=10)
    ax.set_title("Mean Frequency Response per Class", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig02_mean_freq_response.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 3: Class distribution bar chart ────────────────────────────────────
def plot_class_distribution(labels):
    counts = [np.sum(labels == i) for i in range(len(CLASS_NAMES))]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=PALETTE, edgecolor="black", linewidth=0.6)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(count),
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("System Class", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=10)
    ax.set_title("Dataset Class Distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig03_class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 4: Frequency response heatmap per class ────────────────────────────
def plot_freq_response_heatmap(freq_resp, labels):
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(14, 3), sharey=True)
    fig.suptitle("Frequency Response Heatmap per Class (200 samples)", fontsize=11, fontweight="bold")
    freqs = np.linspace(0, np.pi, freq_resp.shape[1])

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        mask = np.where(labels == cls_idx)[0][:200]
        sub = freq_resp[mask]
        im = ax.imshow(
            sub, aspect="auto", origin="lower",
            extent=[freqs[0], freqs[-1], 0, len(mask)],
            cmap="viridis", vmin=0, vmax=sub.max() if sub.max() > 0 else 1,
        )
        ax.set_title(cls_name, fontsize=9)
        ax.set_xlabel("Freq (rad)", fontsize=8)
        if cls_idx == 0:
            ax.set_ylabel("Sample index", fontsize=8)

    plt.colorbar(im, ax=axes[-1], label="Magnitude")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig04_freq_response_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 5: Pixel intensity distribution ────────────────────────────────────
def plot_pixel_distribution(images, labels):
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        pixels = images[mask].flatten()
        ax.hist(pixels, bins=50, alpha=0.5, label=cls_name,
                color=PALETTE[cls_idx], density=True)
    ax.set_xlabel("Pixel Intensity", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Pixel Intensity Distribution of Pole-Zero Map Images", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig05_pixel_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 6: Pole-zero scatter on Z-plane (all classes overlaid) ─────────────
def plot_pz_scatter(poles_vec, zeros_vec, labels, max_per_class=300):
    max_pz = poles_vec.shape[1] // 2
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(14, 3))
    fig.suptitle("Pole-Zero Scatter on Z-Plane per Class", fontsize=11, fontweight="bold")
    theta = np.linspace(0, 2 * np.pi, 300)

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        idxs = np.where(labels == cls_idx)[0][:max_per_class]
        pv = poles_vec[idxs]
        zv = zeros_vec[idxs]

        pole_real = pv[:, 0::2].flatten()
        pole_imag = pv[:, 1::2].flatten()
        zero_real = zv[:, 0::2].flatten()
        zero_imag = zv[:, 1::2].flatten()

        # Filter out zero-padded entries
        pm = (pole_real != 0) | (pole_imag != 0)
        zm = (zero_real != 0) | (zero_imag != 0)

        ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5, alpha=0.5)
        ax.scatter(pole_real[pm], pole_imag[pm], s=3, c=PALETTE[cls_idx],
                   marker="x", alpha=0.4, label="poles")
        ax.scatter(zero_real[zm], zero_imag[zm], s=3, c="gray",
                   marker="o", alpha=0.3, label="zeros")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_title(cls_name, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
        ax.axhline(0, color="k", linewidth=0.3)
        ax.axvline(0, color="k", linewidth=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig06_pz_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def main():
    print("[INFO] Loading dataset ...")
    images, freq_resp, poles_vec, zeros_vec, labels = load_dataset()
    print(f"[INFO] Dataset loaded: {len(images)} samples, {len(CLASS_NAMES)} classes.")

    plot_sample_pz_maps(images, labels)
    plot_mean_freq_response(freq_resp, labels)
    plot_class_distribution(labels)
    plot_freq_response_heatmap(freq_resp, labels)
    plot_pixel_distribution(images, labels)
    plot_pz_scatter(poles_vec, zeros_vec, labels)

    print(f"\n[DONE] All dataset visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
