# Facial Expression Recognition — Model Comparison Project

Course project comparing five machine learning approaches to facial
expression recognition on the CK+ (Extended Cohn-Kanade) dataset: KNN,
SVM, a custom CNN, ResNet50 (transfer learning), and EfficientNetB0
(transfer learning).

## 1. Project Structure

```
FER_Project/
├── config.py                     # shared paths and constants
├── requirements.txt
├── metrics_summary.csv           # generated after models are trained
├── data/
│   ├── raw/                      # downloaded CK+ dataset lands here
│   └── processed/                # balanced/augmented + split dataset
├── models/                       # saved model weights (.pkl / .pth)
├── figures/                      # training curves, confusion matrices,
│                                  # comparison charts
├── output/                       # screenshots saved by the live demo
│                                  # (empty until the demo app is used)
└── src/
    ├── 01_download_data.py
    ├── 02_preprocess_data.py
    ├── 03_feature_engineering.py
    ├── 04_model_knn.py
    ├── 05_model_svm.py
    ├── 06_model_cnn.py
    ├── 07_model_resnet50.py
    ├── 08_model_efficientnet.py
    ├── 09_compare_models.py
    ├── 10_live_demo.py
    └── utils.py
```

## 2. Setup

```bash
pip install -r requirements.txt
```

A Kaggle account and API token are required for the download step. Kaggle's
current token format starts with `KGAT_` and is shown on the API settings
page (kaggle.com → Settings → API → Create New Token). Set it up either way:

```bash
# Option A: environment variable
export KAGGLE_API_TOKEN=KGAT_your_token_here

# Option B: saved token file
mkdir -p ~/.kaggle && echo KGAT_your_token_here > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Never commit the token itself to a public repository.

## 3. Running the Pipeline

Run the scripts in order from the project root:

```bash
python src/01_download_data.py
python src/02_preprocess_data.py
python src/03_feature_engineering.py
python src/04_model_knn.py
python src/05_model_svm.py
python src/06_model_cnn.py
python src/07_model_resnet50.py
python src/08_model_efficientnet.py
python src/09_compare_models.py
```

Each model script trains its model, evaluates it on the held-out test
set, saves the trained model into `models/`, saves its own charts into
`figures/`, and appends one row to `metrics_summary.csv`. The comparison
script reads that CSV once all five rows are present and produces the
combined comparison charts and table.

## 4. Live Demo

```bash
python src/10_live_demo.py
```

This opens a small local web app with one button per model. Upload a
face image, click a model button, and that specific model runs
inference on the image. The resulting panel (face + predicted emotion +
per-class confidence bars) is saved into `output/` with the model name
and a timestamp in the filename, for example `ResNet50_20260702_101530.png`.
The same photo can be run through all five buttons to build up a set of
side-by-side captures in `output/`.

Note: `output/` is intentionally empty in this repository. It only
fills up once the live demo app above has actually been run and a model
button has been clicked.

## 5. Dataset

CK+ (Extended Cohn-Kanade), sourced from Kaggle
(`shawon10/ckplus`). Seven emotion classes. The raw dataset is unbalanced
across classes, so the preprocessing step augments every class up to the
same target count (horizontal flip, rotation, color jitter, small affine
shift) before splitting into train / validation / test (80 / 10 / 10,
stratified).

## 6. Models

| Model | Type | Notes |
|---|---|---|
| KNN | Classical ML | k=3, flattened grayscale features |
| SVM | Classical ML | RBF kernel, C=10 |
| CNN | Deep Learning | 4-block conv net, trained from scratch |
| ResNet50 | Deep Learning | ImageNet pretrained, layer4 + head fine-tuned |
| EfficientNetB0 | Deep Learning | ImageNet pretrained, last blocks + head fine-tuned |

## 7. License

For coursework and educational use.

---
---

# 面部表情识别 — 模型对比项目

课程项目，比较五种机器学习方法在 CK+（扩展版 Cohn-Kanade）数据集上对面部
表情的识别效果：KNN、SVM、自建 CNN、ResNet50（迁移学习）、EfficientNetB0
（迁移学习）。

## 1. 项目结构

```
FER_Project/
├── config.py                     # 共用的路径与常量配置
├── requirements.txt
├── metrics_summary.csv           # 训练完成后自动生成
├── data/
│   ├── raw/                      # CK+ 原始数据下载到这里
│   └── processed/                # 均衡增强并划分好的数据集
├── models/                       # 保存的模型权重（.pkl / .pth）
├── figures/                      # 训练曲线、混淆矩阵、对比图表
├── output/                       # 实时演示程序保存的截图
│                                  # （在使用演示程序之前该文件夹为空）
└── src/
    ├── 01_download_data.py
    ├── 02_preprocess_data.py
    ├── 03_feature_engineering.py
    ├── 04_model_knn.py
    ├── 05_model_svm.py
    ├── 06_model_cnn.py
    ├── 07_model_resnet50.py
    ├── 08_model_efficientnet.py
    ├── 09_compare_models.py
    ├── 10_live_demo.py
    └── utils.py
```

## 2. 环境准备

```bash
pip install -r requirements.txt
```

下载数据集需要 Kaggle 账号及 API 密钥。Kaggle 目前使用的密钥格式以
`KGAT_` 开头，在 kaggle.com → Settings → API → Create New Token 页面
可以获取。任选以下一种方式配置即可：

```bash
# 方式一：设置环境变量
export KAGGLE_API_TOKEN=KGAT_你的密钥

# 方式二：保存为密钥文件
mkdir -p ~/.kaggle && echo KGAT_你的密钥 > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

请注意不要把密钥本身提交到公开仓库中。

## 3. 运行流程

在项目根目录下按顺序运行以下脚本：

```bash
python src/01_download_data.py
python src/02_preprocess_data.py
python src/03_feature_engineering.py
python src/04_model_knn.py
python src/05_model_svm.py
python src/06_model_cnn.py
python src/07_model_resnet50.py
python src/08_model_efficientnet.py
python src/09_compare_models.py
```

每个模型脚本都会完成训练、在测试集上评估、把训练好的模型保存到
`models/`、把对应图表保存到 `figures/`，并在 `metrics_summary.csv`
中追加一行结果。等五个模型的结果都齐全后，对比脚本会读取这份 CSV，
生成最终的综合对比图表和表格。

## 4. 实时演示程序

```bash
python src/10_live_demo.py
```

程序会在本地打开一个小型网页应用，界面上有五个模型按钮。上传一张人脸
图片后，点击任意一个模型按钮，该模型就会对图片进行推理。结果面板
（人脸图像 + 预测的表情类别 + 各类别置信度条形图）会被保存到
`output/` 文件夹，文件名包含模型名称和时间戳，例如
`ResNet50_20260702_101530.png`。同一张照片可以依次点击五个按钮，
在 `output/` 中生成五个模型的对比截图。

需要说明的是：本仓库中的 `output/` 文件夹默认是空的，只有实际运行了
上面的演示程序并点击过模型按钮之后，里面才会出现截图文件。

## 5. 数据集说明

数据集为 CK+（扩展版 Cohn-Kanade），来源于 Kaggle
（`shawon10/ckplus`），共七种表情类别。原始数据集各类别数量不均衡，
因此预处理阶段会通过数据增强（水平翻转、旋转、色彩抖动、小幅仿射平移）
把每个类别补充到相同数量，然后按 80 / 10 / 10 的比例分层划分为训练集、
验证集和测试集。

## 6. 模型说明

| 模型 | 类型 | 说明 |
|---|---|---|
| KNN | 传统机器学习 | k=3，使用展平后的灰度图特征 |
| SVM | 传统机器学习 | RBF 核函数，C=10 |
| CNN | 深度学习 | 四层卷积网络，从零开始训练 |
| ResNet50 | 深度学习 | 基于 ImageNet 预训练权重，微调 layer4 与分类头 |
| EfficientNetB0 | 深度学习 | 基于 ImageNet 预训练权重，微调最后几层与分类头 |

## 7. 许可说明

仅用于课程作业与学习用途。
