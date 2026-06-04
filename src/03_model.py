"""
03_model.py
===========
Course  : Digital Image Processing
Major   : Artificial Intelligence
Name    : Hossen Md Jisan
ID      : 202353460019
Topic   : Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

Description
-----------
I define three deep learning models for system identification:

  1. PoleZeroCNN   — CNN that takes the rendered pole-zero map image as input
                     and predicts the system class.
  2. FreqMLP       — MLP that takes the frequency response vector as input.
  3. FusionNet     — Late-fusion model combining CNN (image branch) and
                     MLP (frequency branch) for improved accuracy.

All models output logits over N_CLASSES = 5 system classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES  = 5
IMG_SIZE   = 64
N_FREQ     = 128


# ── Helper: conv block ────────────────────────────────────────────────────────
def conv_block(in_ch, out_ch, kernel=3, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


# ── Model 1: PoleZeroCNN ──────────────────────────────────────────────────────
class PoleZeroCNN(nn.Module):
    """
    CNN for classifying digital systems from their pole-zero map images.
    Input : (B, 1, 64, 64)  grayscale pole-zero map
    Output: (B, N_CLASSES)  logits
    """
    def __init__(self, n_classes=N_CLASSES, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1,  32, pool=True),   # -> (B, 32, 32, 32)
            conv_block(32, 64, pool=True),   # -> (B, 64, 16, 16)
            conv_block(64, 128, pool=True),  # -> (B,128,  8,  8)
            conv_block(128, 256, pool=True), # -> (B,256,  4,  4)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B,256, 1,  1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ── Model 2: FreqMLP ─────────────────────────────────────────────────────────
class FreqMLP(nn.Module):
    """
    MLP for classifying digital systems from their frequency response vectors.
    Input : (B, N_FREQ)     frequency magnitude vector
    Output: (B, N_CLASSES)  logits
    """
    def __init__(self, n_freq=N_FREQ, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Model 3: FusionNet (CNN + MLP) ───────────────────────────────────────────
class FusionNet(nn.Module):
    """
    Late-fusion model combining image branch (CNN) and frequency branch (MLP).
    Input : image  (B, 1, 64, 64)
            freq   (B, N_FREQ)
    Output: (B, N_CLASSES) logits
    """
    def __init__(self, n_freq=N_FREQ, n_classes=N_CLASSES, dropout=0.4):
        super().__init__()

        # ── Image branch ─────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            conv_block(1,  32, pool=True),
            conv_block(32, 64, pool=True),
            conv_block(64, 128, pool=True),
            conv_block(128, 256, pool=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                     # -> (B, 256)
        )

        # ── Frequency branch ─────────────────────────────────────────────────
        self.freq_mlp = nn.Sequential(
            nn.Linear(n_freq, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),            # -> (B, 64)
        )

        # ── Fusion head ──────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, image, freq):
        img_feat  = self.cnn(image)           # (B, 256)
        freq_feat = self.freq_mlp(freq)       # (B,  64)
        fused     = torch.cat([img_feat, freq_feat], dim=1)  # (B, 320)
        return self.head(fused)


# ── Factory function ──────────────────────────────────────────────────────────
def build_model(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.
    Valid names: 'cnn', 'mlp', 'fusion'
    """
    name = name.lower()
    if name == "cnn":
        return PoleZeroCNN(**kwargs)
    elif name in ("mlp", "freq_mlp"):
        return FreqMLP(**kwargs)
    elif name == "fusion":
        return FusionNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}. Choose from cnn / mlp / fusion.")


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    B = 4
    img  = torch.randn(B, 1, IMG_SIZE, IMG_SIZE)
    freq = torch.randn(B, N_FREQ)

    for name in ["cnn", "mlp", "fusion"]:
        if name == "fusion":
            model = build_model(name)
            out = model(img, freq)
        elif name == "cnn":
            model = build_model(name)
            out = model(img)
        else:
            model = build_model(name)
            out = model(freq)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{name.upper():8s}] output shape: {out.shape}  |  params: {total_params:,}")

    print("[DONE] Model definitions verified.")
