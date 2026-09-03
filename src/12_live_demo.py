# -*- coding: utf-8 -*-
"""
12_live_demo.py

This is the interactive part of the project. I built a small Gradio app
with one button per model (KNN, SVM, CNN, VGG16, MobileNetV2, ResNet50,
EfficientNetB0). I upload a face image, pick a button, and that specific
model runs inference on it. The app then renders a result panel (the face
plus the predicted emotion and confidence bars for every class) and saves
that panel as a PNG into the output/ folder, named with the model and a
timestamp so repeated runs never overwrite each other.

I kept the seven buttons completely independent on purpose, that way I can
run the exact same photo through all seven models one at a time and
compare the saved output images side by side afterward.

Before running this, every model needs to already be trained and saved
(or just run train_all.py once):
    python src/04_model_knn.py
    python src/05_model_svm.py
    python src/06_model_cnn.py
    python src/07_model_vgg16.py
    python src/08_model_mobilenetv2.py
    python src/09_model_resnet50.py
    python src/10_model_efficientnet.py

Then:
    python src/12_live_demo.py
"""

import os
import sys
import io
import pickle
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, OUTPUT_DIR, EMOTION_LABELS, IMG_SIZE_FLAT, ensure_dirs
from utils import get_eval_transform, get_device

ensure_dirs()
device = get_device()
eval_transform = get_eval_transform()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ── Face cropping helper, shared by every model's prediction path ──────────
def crop_face(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad = int(0.15 * w)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image_np.shape[1], x + w + pad), min(image_np.shape[0], y + h + pad)
        face_gray = gray[y1:y2, x1:x2]
        face_rgb = image_np[y1:y2, x1:x2]
    else:
        face_gray = gray
        face_rgb = image_np
    return face_rgb, face_gray


# ── Lazy model loading, I only load a model into memory the first time its
# button gets clicked, no need to hold all five in RAM if I'm only testing
# one or two of them right now ──────────────────────────────────────────────
_loaded = {}


def build_cnn_arch():
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(4)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, len(EMOTION_LABELS))
            )

        def forward(self, x):
            return self.classifier(self.features(x))
    return SimpleCNN(len(EMOTION_LABELS))


def build_resnet_arch():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, len(EMOTION_LABELS))
    )
    return model


def build_effnet_arch():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(EMOTION_LABELS))
    return model


def build_vgg_arch():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, len(EMOTION_LABELS))
    )
    return model


def build_mobilenet_arch():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(EMOTION_LABELS))
    return model


def load_model(model_name):
    if model_name in _loaded:
        return _loaded[model_name]

    if model_name == 'KNN':
        path = os.path.join(MODELS_DIR, 'knn_model.pkl')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    elif model_name == 'SVM':
        path = os.path.join(MODELS_DIR, 'svm_model.pkl')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    elif model_name == 'CNN':
        obj = build_cnn_arch()
        obj.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'CNN_best.pth'), map_location=device))
        obj.to(device).eval()
    elif model_name == 'ResNet50':
        obj = build_resnet_arch()
        obj.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ResNet50_best.pth'), map_location=device))
        obj.to(device).eval()
    elif model_name == 'EfficientNetB0':
        obj = build_effnet_arch()
        obj.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'EfficientNetB0_best.pth'), map_location=device))
        obj.to(device).eval()
    elif model_name == 'VGG16':
        obj = build_vgg_arch()
        obj.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'VGG16_best.pth'), map_location=device))
        obj.to(device).eval()
    elif model_name == 'MobileNetV2':
        obj = build_mobilenet_arch()
        obj.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'MobileNetV2_best.pth'), map_location=device))
        obj.to(device).eval()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    _loaded[model_name] = obj
    return obj


# ── Prediction paths, classical ML vs deep models need different inputs ────
def predict_classical(model_name, face_gray):
    scaler_path = os.path.join(MODELS_DIR, '..', 'data', 'processed', 'flat_features', 'scaler.pkl')
    scaler_path = os.path.normpath(scaler_path)
    model = load_model(model_name)

    small = cv2.resize(face_gray, (IMG_SIZE_FLAT, IMG_SIZE_FLAT))
    flat = small.astype(np.float32).flatten().reshape(1, -1)

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        flat = scaler.transform(flat)

    probs = model.predict_proba(flat)[0]
    pred_idx = int(np.argmax(probs))
    return EMOTION_LABELS[pred_idx], probs


def predict_deep(model_name, face_rgb):
    model = load_model(model_name)
    tensor = eval_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return EMOTION_LABELS[pred_idx], probs


def run_prediction(model_name, image_np):
    face_rgb, face_gray = crop_face(image_np)
    if model_name in ('KNN', 'SVM'):
        return predict_classical(model_name, face_gray), face_rgb
    else:
        return predict_deep(model_name, face_rgb), face_rgb


# ── Result panel rendering + screenshot saving ──────────────────────────────
def render_result_panel(model_name, face_rgb, pred_label, probs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), facecolor='white')
    fig.suptitle(f'{model_name} — Prediction: {pred_label.capitalize()}',
                 fontsize=13, fontweight='bold')

    axes[0].imshow(face_rgb)
    axes[0].axis('off')
    axes[0].set_title('Input Face', fontsize=10)

    colors = ['#27ae60' if l == pred_label else '#bdc3c7' for l in EMOTION_LABELS]
    axes[1].barh(EMOTION_LABELS, probs * 100, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Class Probabilities', fontsize=10)
    for i, v in enumerate(probs * 100):
        if v > 2:
            axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    panel_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return panel_img


def save_screenshot(model_name, panel_img):
    """This is where I write the result panel into output/ using the model
    name in the filename, exactly what I need to compare captures across
    models after running the same photo through all five buttons."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{model_name}_{timestamp}.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    panel_img.save(save_path)
    return save_path


def analyze_with_model(model_name, image):
    if image is None:
        return None, 'Please upload an image first.'

    image_np = np.array(image.convert('RGB'))
    model_path_map = {
        'KNN': 'knn_model.pkl', 'SVM': 'svm_model.pkl',
        'CNN': 'CNN_best.pth', 'VGG16': 'VGG16_best.pth',
        'MobileNetV2': 'MobileNetV2_best.pth',
        'ResNet50': 'ResNet50_best.pth',
        'EfficientNetB0': 'EfficientNetB0_best.pth',
    }
    expected_file = os.path.join(MODELS_DIR, model_path_map[model_name])
    if not os.path.exists(expected_file):
        msg = (f'I could not find a saved {model_name} model at {expected_file}. '
               f'Run the matching training script first.')
        return None, msg

    (pred_label, probs), face_rgb = run_prediction(model_name, image_np)
    panel_img = render_result_panel(model_name, face_rgb, pred_label, probs)
    save_path = save_screenshot(model_name, panel_img)

    status = (f'{model_name} predicts: {pred_label.upper()} '
              f'({probs.max()*100:.1f}% confidence)\n'
              f'Saved capture to: {save_path}')
    return panel_img, status


def build_app():
    import gradio as gr

    with gr.Blocks(title='FER Live Model Comparison') as demo:
        gr.Markdown(
            "# Facial Expression Recognition — Live Model Comparison\n"
            "Upload a face image, then click any of the seven model buttons "
            "below. Each click runs that specific model and saves the "
            "result panel into the output folder, named with the model."
        )

        image_input = gr.Image(label='Upload a face image', type='pil', height=320)

        with gr.Row():
            btn_knn = gr.Button('KNN')
            btn_svm = gr.Button('SVM')
            btn_cnn = gr.Button('CNN')
            btn_vgg = gr.Button('VGG16')
            btn_mobilenet = gr.Button('MobileNetV2')
            btn_resnet = gr.Button('ResNet50')
            btn_effnet = gr.Button('EfficientNetB0')

        result_image = gr.Image(label='Result Panel', type='pil', height=360)
        status_box = gr.Textbox(label='Status', lines=3)

        btn_knn.click(fn=lambda img: analyze_with_model('KNN', img),
                       inputs=image_input, outputs=[result_image, status_box])
        btn_svm.click(fn=lambda img: analyze_with_model('SVM', img),
                       inputs=image_input, outputs=[result_image, status_box])
        btn_cnn.click(fn=lambda img: analyze_with_model('CNN', img),
                       inputs=image_input, outputs=[result_image, status_box])
        btn_vgg.click(fn=lambda img: analyze_with_model('VGG16', img),
                       inputs=image_input, outputs=[result_image, status_box])
        btn_mobilenet.click(fn=lambda img: analyze_with_model('MobileNetV2', img),
                             inputs=image_input, outputs=[result_image, status_box])
        btn_resnet.click(fn=lambda img: analyze_with_model('ResNet50', img),
                          inputs=image_input, outputs=[result_image, status_box])
        btn_effnet.click(fn=lambda img: analyze_with_model('EfficientNetB0', img),
                          inputs=image_input, outputs=[result_image, status_box])

    return demo


if __name__ == '__main__':
    app = build_app()
    app.launch(share=False, debug=False)
