# FER_2026 — Facial Expression Recognition

Comparative study of 5 deep learning models for facial expression recognition. Trained and fine-tuned simultaneously to identify the most efficient and accurate architecture.

## Models Evaluated

1. **VGG16** - Transfer learning with custom classifier
2. **ResNet50** - Skip connections for deep architectures
3. **MobileNetV2** - Lightweight, mobile-optimized
4. **EfficientNetB0** - Compound scaling approach
5. **Custom CNN** - Purpose-built architecture

## Dataset

- **Source**: FER-2013 / Custom augmented dataset
- **Classes**: 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Split**: 80% train, 10% validation, 10% test

## Results

| Model | Accuracy | Params | Inference Time |
|-------|----------|--------|----------------|
| VGG16 | TBD | 14.7M | TBD ms |
| ResNet50 | TBD | 23.5M | TBD ms |
| MobileNetV2 | TBD | 2.2M | TBD ms |
| EfficientNetB0 | TBD | 4.0M | TBD ms |
| Custom CNN | TBD | 0.5M | TBD ms |

## Tech Stack

- **Framework**: PyTorch / TensorFlow
- **Training**: GPU-accelerated (CUDA)
- **Visualization**: Matplotlib, TensorBoard
- **Preprocessing**: OpenCV, Albumentations

## Quick Start

```bash
pip install -r requirements.txt

# Train all models
python train_all_models.py

# Evaluate
python evaluate.py --model resnet50

# Inference on single image
python predict.py --image path/to/face.jpg
```

## License

MIT
