"""
backend/app/inference.py

Loads the trained MobileNetV2 model ONCE at import time
and exposes a predict() function.

Why load once: loading the model takes ~2 seconds. We don't want
to do that every request — we'd never hit the 3-second KPI.
"""
import torch
import json
from torch import nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
from pathlib import Path

_HERE = Path(__file__).parent

# Dell laptop = no GPU, just CPU
DEVICE = torch.device('cpu')

# Load class labels
with open(_HERE / 'labels.json') as f:
    LABELS = json.load(f)

print(f"[inference] Loading model with {len(LABELS)} classes...")

# Rebuild the model architecture
_model = models.mobilenet_v2(weights=None)
_model.classifier[1] = nn.Linear(_model.classifier[1].in_features, len(LABELS))

# Load the trained weights
_model.load_state_dict(torch.load(_HERE / 'model.pth', map_location=DEVICE))
_model = _model.to(DEVICE).eval()

print("[inference] Model loaded and ready.")

# Image preprocessing — MUST match training exactly
_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image_bytes: bytes, top_k: int = 3):
    """
    Predict the top-k clothing categories for an image.

    Args:
        image_bytes: raw bytes of the uploaded image (JPEG or PNG)
        top_k: how many predictions to return

    Returns:
        List of (label, confidence) tuples, e.g. [('Tshirts', 0.92), ...]
    """
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    x = _tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = _model(x)
        probs = torch.softmax(out, dim=1)[0]

    top_probs, top_idx = probs.topk(top_k)
    return [(LABELS[i], float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())]
