"""
06_run_all.py
=============
Course  : Digital Image Processing
Major   : Artificial Intelligence
Name    : Hossen Md Jisan
ID      : 202353460019
Topic   : Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

Description
-----------
I run the full pipeline end-to-end in one command:
  Step 1  — Dataset generation
  Step 2  — Dataset visualization
  Step 3  — Model verification (architecture print)
  Step 4  — Training (CNN, MLP, FusionNet)
  Step 5  — Evaluation (metrics, figures, test report)

All outputs are auto-saved to output/ and dataset/.
"""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")

STEPS = [
    ("01  Dataset Generation",      "01_dataset_generation.py"),
    ("02  Dataset Visualization",   "02_dataset_visualization.py"),
    ("03  Model Verification",      "03_model.py"),
    ("04  Model Training",          "04_train.py"),
    ("05  Model Evaluation",        "05_evaluate.py"),
]


def banner(text):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def run_step(label, script):
    banner(f"STEP: {label}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, script)],
        cwd=BASE_DIR,
    )
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {label}  ({elapsed:.1f}s)")
    if result.returncode != 0:
        print(f"  [ERROR] Script {script} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main():
    banner("DIP — Z-Transform Pole-Zero Deep Learning Pipeline")
    print(f"  Name      : Hossen Md Jisan")
    print(f"  ID        : 202353460019")
    print(f"  Course    : Digital Image Processing")
    print(f"  Base Dir  : {BASE_DIR}")

    total_start = time.time()
    for label, script in STEPS:
        run_step(label, script)

    total = time.time() - total_start
    banner(f"ALL STEPS COMPLETE  ({total:.1f}s total)")
    print("  Outputs saved to:")
    print(f"    dataset/              — .npz, .csv, checkpoints")
    print(f"    output/figures/       — training curves, confusion matrices, predictions")
    print(f"    output/dataset_viz/   — pole-zero maps, frequency response plots")
    print(f"    output/test_metrics/  — PSNR, SSIM, F1, accuracy (CSV + JSON + PNG)")


if __name__ == "__main__":
    main()
