# 基于Z变换极点与零点的深度学习系统辨识

**课程名称：** 数字图像处理
**专业：** 人工智能
**姓名：** Hossen Md Jisan
**学号：** 202353460019
**研究主题：** 基于Z变换极点与零点的深度学习系统辨识

---

## 项目概述

在本项目中，我提出了一种基于深度学习的离散时间LTI系统辨识框架。我将每个系统的Z变换极零点分布图渲染为64x64的灰度图像，并结合128维频率响应幅度向量，构建三种神经网络架构：卷积网络PoleZeroCNN、多层感知机FreqMLP和后期融合网络FusionNet，将系统分类为低通、高通、带通、带阻与全通五类。

该框架将经典数字信号处理与现代深度学习相结合，在数字图像处理领域的滤波器识别、图像去噪、盲目反卷积等任务中具有直接应用价值。

---

## 实验结果

我在5000个样本（每类1000个）的均衡数据集上训练并评估三个模型，测试集750个样本：

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 参数量 |
|------|--------|--------|--------|--------|--------|
| PoleZeroCNN | 99.73% | 99.76% | 99.73% | 99.74% | 421,861 |
| FreqMLP | 100.00% | 100.00% | 100.00% | 100.00% | 75,269 |
| FusionNet | 100.00% | 100.00% | 100.00% | 100.00% | 455,077 |

---

## 目录结构

```
DIP_ZTransform_DL/
├── src/
│   ├── 01_dataset_generation.py      # 合成极零点数据集（5000样本，5类）
│   ├── 02_dataset_visualization.py   # 数据集可视化（6张图自动保存）
│   ├── 03_model.py                   # CNN、MLP、FusionNet架构定义
│   ├── 04_train.py                   # 训练、检查点、学习率调度
│   ├── 05_evaluate.py                # 测试评估、混淆矩阵、指标输出
│   └── 06_run_all.py                 # 一键运行完整流程
├── dataset/                          # 运行后自动生成
├── output/
│   ├── figures/                      # 训练曲线、混淆矩阵、预测面板（12张）
│   ├── dataset_viz/                  # 数据集可视化（6张）
│   └── test_metrics/                 # 测试指标CSV、JSON、对比图（共19张PNG）
├── paper/
│   ├── main.tex                      # LaTeX论文完整源码
│   └── figures/                      # 论文图像目录（从output/复制）
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 环境配置

我建议使用Python 3.9及以上版本：

```bash
pip install -r requirements.txt
```

主要依赖：PyTorch 2.0+、NumPy、SciPy、Matplotlib、scikit-learn、scikit-image、Pillow、pandas、seaborn、tqdm。

---

## 运行方式

```bash
# 一键运行（推荐）
python src/06_run_all.py

# 分步运行
python src/01_dataset_generation.py   # 约96秒（CPU）
python src/02_dataset_visualization.py
python src/03_model.py                # 架构验证
python src/04_train.py                # 约110秒（GPU）
python src/05_evaluate.py
```

在GPU环境下完整流程约耗时224秒。

---

## 输出文件（共19张PNG）

**output/dataset_viz/**（6张）
- fig01_sample_pz_maps.png
- fig02_mean_freq_response.png
- fig03_class_distribution.png
- fig04_freq_response_heatmap.png
- fig05_pixel_distribution.png
- fig06_pz_scatter.png

**output/figures/**（12张）
- fig_train_curves_cnn.png
- fig_train_curves_mlp.png
- fig_train_curves_fusion.png
- fig_confusion_cnn.png
- fig_confusion_mlp.png
- fig_confusion_fusion.png
- fig_f1_per_class_cnn.png
- fig_f1_per_class_mlp.png
- fig_f1_per_class_fusion.png
- fig_pz_predictions_cnn.png
- fig_pz_predictions_mlp.png
- fig_pz_predictions_fusion.png

**output/test_metrics/**（1张）
- fig_model_comparison.png

---

## 论文编译

`paper/main.tex` 包含完整LaTeX论文（1257行），结构如下：

- 摘要（含定量结果）
- 引言（背景、问题定义、研究目标）
- 相关工作（2018-2022年文献）
- 数据集构建（Z变换原理、共轭对生成、两种特征表示）
- 模型架构（ConvBlock公式、PoleZeroCNN逐层规格、FreqMLP逐层规格、FusionNet架构图与数学推导、训练目标与优化策略）
- 实验与讨论（全部19张图、5张数据表、误分析）
- 结论与未来工作
- 参考文献（18篇，均为2015-2022年期刊与会议论文）

编译步骤：

```bash
# 第一步：复制图像到论文目录
mkdir -p paper/figures
cp output/dataset_viz/*.png paper/figures/
cp output/figures/*.png paper/figures/
cp output/test_metrics/fig_model_comparison.png paper/figures/

# 第二步：编译（运行两次生成正确引用编号）
cd paper
pdflatex main.tex
pdflatex main.tex
```

---

## 注意事项

- dataset/ 中的 .npz 和 .pt 文件已通过 .gitignore 排除，在本地运行代码后自动生成。
- 若无GPU，代码自动回退到CPU，训练时间相应增加。
- 所有随机种子固定为42，确保完全可复现。
- 论文图像引用路径为 figures/，编译前必须先执行上述复制命令。

---

---

# Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

**Course:** Digital Image Processing
**Major:** Artificial Intelligence
**Name:** Hossen Md Jisan
**Student ID:** 202353460019
**Topic:** Deep Learning-Based System Identification Using Z-Transform Poles and Zeros

---

## Project Overview

In this project, I propose a deep learning framework for discrete-time LTI system identification using Z-transform pole-zero representations. I render each system's pole-zero map as a 64x64 grayscale image and additionally extract a 128-dimensional frequency-response magnitude vector. I design and compare three neural architectures: PoleZeroCNN (image-only CNN), FreqMLP (frequency-vector MLP), and FusionNet (late-fusion of both modalities). Systems are classified into five frequency-selective categories: lowpass, highpass, bandpass, bandstop, and allpass.

This work bridges classical digital signal processing theory with modern deep learning and demonstrates that Z-transform pole-zero geometry is a rich, learnable representation for automated filter recognition, with direct applications to image denoising, blind deconvolution, and adaptive image restoration.

---

## Results Summary

Trained on 5,000 balanced samples (1,000 per class) and evaluated on 750 held-out test samples:

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| PoleZeroCNN | 99.73% | 99.76% | 99.73% | 99.74% | 421,861 |
| FreqMLP | 100.00% | 100.00% | 100.00% | 100.00% | 75,269 |
| FusionNet | 100.00% | 100.00% | 100.00% | 100.00% | 455,077 |

---

## Repository Structure

```
DIP_ZTransform_DL/
├── src/
│   ├── 01_dataset_generation.py      # Synthetic pole-zero dataset (5000 samples, 5 classes)
│   ├── 02_dataset_visualization.py   # Dataset visualization (6 figures auto-saved)
│   ├── 03_model.py                   # CNN, MLP, FusionNet model definitions
│   ├── 04_train.py                   # Training loop, checkpoints, LR scheduling
│   ├── 05_evaluate.py                # Test evaluation, confusion matrices, metrics
│   └── 06_run_all.py                 # One-command full pipeline
├── dataset/                          # Auto-populated on run
├── output/
│   ├── figures/                      # Training curves, confusion matrices, predictions (12 PNG)
│   ├── dataset_viz/                  # Dataset visualizations (6 PNG)
│   └── test_metrics/                 # Metrics CSV, JSON, comparison chart (1 PNG)
├── paper/
│   ├── main.tex                      # Complete LaTeX paper source (1257 lines)
│   └── figures/                      # Paper figures directory (copy from output/)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

Python 3.9 or higher recommended:

```bash
pip install -r requirements.txt
```

Core dependencies: PyTorch 2.0+, NumPy, SciPy, Matplotlib, scikit-learn, scikit-image, Pillow, pandas, seaborn, tqdm.

---

## How to Run

```bash
# Option 1: One-command full pipeline (recommended)
python src/06_run_all.py

# Option 2: Step by step
python src/01_dataset_generation.py   # ~96s on CPU
python src/02_dataset_visualization.py
python src/03_model.py                # Architecture verification
python src/04_train.py                # ~110s on GPU
python src/05_evaluate.py
```

Full pipeline completes in approximately 224 seconds on GPU.

---

## Output Files (19 PNG total)

**output/dataset_viz/** (6 figures)
- fig01_sample_pz_maps.png - Sample pole-zero map images per class
- fig02_mean_freq_response.png - Mean frequency response per class
- fig03_class_distribution.png - Class balance bar chart
- fig04_freq_response_heatmap.png - Frequency response heatmap
- fig05_pixel_distribution.png - Pixel intensity distributions
- fig06_pz_scatter.png - Pole-zero scatter on Z-plane

**output/figures/** (12 figures)
- fig_train_curves_{cnn,mlp,fusion}.png - Training and validation curves
- fig_confusion_{cnn,mlp,fusion}.png - Normalized confusion matrices
- fig_f1_per_class_{cnn,mlp,fusion}.png - Per-class F1 score bar charts
- fig_pz_predictions_{cnn,mlp,fusion}.png - Prediction panels (GT vs predicted)

**output/test_metrics/** (1 figure)
- fig_model_comparison.png - Side-by-side model comparison

---

## Paper Compilation

The file paper/main.tex contains the complete LaTeX source (1,257 lines) structured per course requirements:

- Abstract (with quantitative results)
- Introduction (background, problem statement, research objectives, paper organization)
- Related Work (classical system ID, deep learning for signals, multi-modal fusion, image quality assessment - all references 2015-2022)
- Dataset Construction (Z-transform theory, conjugate pair generation, class placement rules with worked examples, Algorithm box, two feature representations)
- Model Architectures (shared ConvBlock derivation; PoleZeroCNN with layer table and full forward-pass equations; FreqMLP with layer table and forward-pass; FusionNet with TikZ architecture diagram, mathematical forward pass, and parameter table; training objective with label smoothing; Adam optimizer and cosine annealing equations)
- Experiments and Discussion (all 19 PNG figures, 5 data tables, quantitative analysis, lowpass-allpass confusion explanation, PSNR/SSIM interpretation, relevance to image processing)
- Conclusion and Future Work
- References (18 references, 2015-2022)

To compile:

```bash
# Step 1: Copy figures into paper directory
mkdir -p paper/figures
cp output/dataset_viz/*.png paper/figures/
cp output/figures/*.png paper/figures/
cp output/test_metrics/fig_model_comparison.png paper/figures/

# Step 2: Compile (run twice for correct cross-references)
cd paper
pdflatex main.tex
pdflatex main.tex
```

---

## Notes

- The .npz and .pt files in dataset/ are excluded from git via .gitignore due to their size. They are regenerated automatically by running the pipeline.
- The code falls back to CPU automatically if no GPU is available.
- All random seeds are fixed at 42 for full reproducibility.
- Figure paths in the LaTeX source are relative to paper/figures/. The copy step above must be completed before compilation.
