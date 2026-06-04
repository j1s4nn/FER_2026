"""
01_dataset_generation.py
========================
Course  : Digital Image Processing
Major   : Artificial Intelligence
Name    : Hossen Md Jisan
ID      : 202353460019
Topic   : Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

Description
-----------
I generate a synthetic dataset of discrete-time LTI systems characterized by their
Z-transform poles and zeros. Each sample contains:
  - Pole locations (real + imaginary) on the Z-plane
  - Zero locations (real + imaginary) on the Z-plane
  - Rendered pole-zero map image (64x64 grayscale)
  - Frequency response magnitude (as 1D feature vector)
  - System class label (lowpass / highpass / bandpass / bandstop / allpass)

All data is saved to the dataset/ directory as .npz and .csv files.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import freqz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
N_SAMPLES    = 5000          # total dataset size
IMG_SIZE     = 64            # pole-zero map image resolution
N_FREQ_BINS  = 128           # frequency response vector length
MAX_ORDER    = 4             # maximum filter order (poles / zeros each)
CLASS_NAMES  = ["lowpass", "highpass", "bandpass", "bandstop", "allpass"]
N_CLASSES    = len(CLASS_NAMES)


def make_conjugate_pair(r_min=0.1, r_max=0.95, angle_range=None):
    """Return a conjugate pole/zero pair with radius in [r_min, r_max]."""
    r = np.random.uniform(r_min, r_max)
    if angle_range is None:
        angle = np.random.uniform(0.05, np.pi - 0.05)
    else:
        angle = np.random.uniform(*angle_range)
    z = r * np.exp(1j * angle)
    return [z, np.conj(z)]


def generate_system_by_class(label):
    """
    Generate pole and zero lists for a given filter class.
    Returns (zeros_list, poles_list) as lists of complex numbers.
    """
    order = np.random.randint(1, MAX_ORDER + 1)

    if label == 0:        # lowpass — poles near DC (angle ~ 0), zeros near pi
        poles = []
        zeros = []
        for _ in range(order):
            poles += make_conjugate_pair(0.4, 0.92, (0.02, 0.4))
            zeros += make_conjugate_pair(0.7, 0.99, (2.5, 3.1))

    elif label == 1:      # highpass — poles near pi, zeros near DC
        poles = []
        zeros = []
        for _ in range(order):
            poles += make_conjugate_pair(0.4, 0.92, (2.5, 3.1))
            zeros += make_conjugate_pair(0.7, 0.99, (0.02, 0.4))

    elif label == 2:      # bandpass — poles near mid-band angle
        poles = []
        zeros = []
        for _ in range(order):
            angle = np.random.uniform(0.8, 2.3)
            r = np.random.uniform(0.5, 0.93)
            z = r * np.exp(1j * angle)
            poles += [z, np.conj(z)]
            zeros += [complex(1, 0), complex(-1, 0)]  # zeros at ±1

    elif label == 3:      # bandstop — zeros on unit circle at mid-band
        poles = []
        zeros = []
        for _ in range(order):
            angle = np.random.uniform(0.8, 2.3)
            zeros += [np.exp(1j * angle), np.conj(np.exp(1j * angle))]
            poles += make_conjugate_pair(0.3, 0.7, (angle - 0.15, angle + 0.15))

    else:                 # allpass — poles inside unit circle, zeros reciprocal
        poles = []
        zeros = []
        for _ in range(order):
            r = np.random.uniform(0.3, 0.85)
            angle = np.random.uniform(0.2, np.pi - 0.2)
            p = r * np.exp(1j * angle)
            z_val = (1.0 / np.conj(p))
            poles += [p, np.conj(p)]
            zeros += [z_val, np.conj(z_val)]

    return zeros, poles


def poles_zeros_to_tf(zeros, poles, k=1.0):
    """Convert pole-zero lists to transfer function coefficients b, a."""
    from numpy.polynomial.polynomial import polyfromroots
    def poly(roots):
        if len(roots) == 0:
            return np.array([1.0])
        p = np.poly(roots)     # numpy.poly returns highest-degree first
        return np.real(p)
    b = k * poly(zeros)
    a = poly(poles)
    return b, a


def render_pz_map(zeros, poles, size=IMG_SIZE):
    """
    Render a grayscale pole-zero map image of shape (size, size).
    Poles are drawn as X marks, zeros as O marks.
    """
    fig, ax = plt.subplots(figsize=(2, 2), dpi=size // 2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.3)
    ax.axvline(0, color="k", linewidth=0.3)

    for z in zeros:
        ax.plot(z.real, z.imag, "ko", markersize=3, markerfacecolor="none", markeredgewidth=0.8)
    for p in poles:
        ax.plot(p.real, p.imag, "kx", markersize=3, markeredgewidth=0.8)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    # Compatible with both old and new matplotlib versions
    from PIL import Image
    try:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        gray = 0.2989 * buf[:, :, 0] + 0.5870 * buf[:, :, 1] + 0.1140 * buf[:, :, 2]
    except AttributeError:
        # matplotlib >= 3.8: use buffer_rgba instead
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        gray = 0.2989 * buf[:, :, 0] + 0.5870 * buf[:, :, 1] + 0.1140 * buf[:, :, 2]
    img = Image.fromarray(gray.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    plt.close(fig)
    return np.array(img, dtype=np.float32) / 255.0


def compute_freq_response(b, a, n_bins=N_FREQ_BINS):
    """Return magnitude of frequency response at n_bins uniformly-spaced freqs."""
    try:
        w, h = freqz(b, a, worN=n_bins, fs=2 * np.pi)
        mag = np.abs(h).astype(np.float32)
        # Clip extreme values for numerical stability
        mag = np.clip(mag, 0, 50)
        return mag
    except Exception:
        return np.zeros(n_bins, dtype=np.float32)


def flatten_complex(arr, max_len):
    """Flatten complex array to real vector of length 2*max_len, zero-padded."""
    vec = np.zeros(2 * max_len, dtype=np.float32)
    for i, c in enumerate(arr[:max_len]):
        vec[2 * i]     = float(c.real)
        vec[2 * i + 1] = float(c.imag)
    return vec


def generate_dataset():
    print("[INFO] Generating dataset ...")
    max_pz = MAX_ORDER * 2   # max number of poles or zeros

    images      = np.zeros((N_SAMPLES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    freq_resp   = np.zeros((N_SAMPLES, N_FREQ_BINS),        dtype=np.float32)
    poles_vec   = np.zeros((N_SAMPLES, 2 * max_pz),          dtype=np.float32)
    zeros_vec   = np.zeros((N_SAMPLES, 2 * max_pz),          dtype=np.float32)
    labels      = np.zeros(N_SAMPLES,                         dtype=np.int64)

    meta_rows = []

    for i in tqdm(range(N_SAMPLES), desc="Samples"):
        label = i % N_CLASSES   # balanced classes
        zeros, poles = generate_system_by_class(label)
        b, a = poles_zeros_to_tf(zeros, poles)

        img  = render_pz_map(zeros, poles)
        freq = compute_freq_response(b, a)
        pv   = flatten_complex(poles, max_pz)
        zv   = flatten_complex(zeros, max_pz)

        images[i]    = img
        freq_resp[i] = freq
        poles_vec[i] = pv
        zeros_vec[i] = zv
        labels[i]    = label

        meta_rows.append({
            "sample_id": i,
            "label": label,
            "class_name": CLASS_NAMES[label],
            "n_poles": len(poles),
            "n_zeros": len(zeros),
        })

    # Save numpy archive
    npz_path = os.path.join(DATASET_DIR, "dataset.npz")
    np.savez_compressed(
        npz_path,
        images=images,
        freq_resp=freq_resp,
        poles_vec=poles_vec,
        zeros_vec=zeros_vec,
        labels=labels,
    )
    print(f"[INFO] Dataset saved to {npz_path}")

    # Save metadata CSV
    csv_path = os.path.join(DATASET_DIR, "metadata.csv")
    pd.DataFrame(meta_rows).to_csv(csv_path, index=False)
    print(f"[INFO] Metadata saved to {csv_path}")

    # Class distribution summary
    dist_path = os.path.join(DATASET_DIR, "class_distribution.csv")
    dist = pd.DataFrame(meta_rows).groupby("class_name").size().reset_index(name="count")
    dist.to_csv(dist_path, index=False)
    print(f"[INFO] Class distribution saved to {dist_path}")
    print(dist.to_string(index=False))

    return npz_path


if __name__ == "__main__":
    generate_dataset()
    print("[DONE] Dataset generation complete.")
