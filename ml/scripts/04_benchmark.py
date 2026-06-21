"""
ml/scripts/04_benchmark.py

Measures inference speed on YOUR Dell's CPU. This tells you whether
the <3 second response time KPI from the proposal is achievable.

Usage:
    python 04_benchmark.py
"""
import torch
import time
import json
from torchvision import transforms, models
from torch import nn
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / 'models' / 'fitcheck_mobilenet.pth'
LABELS_PATH = SCRIPT_DIR.parent / 'models' / 'labels.json'
TEST_DIR = SCRIPT_DIR.parent / 'data' / 'processed' / 'test'

device = torch.device('cpu')

with open(LABELS_PATH) as f:
    classes = json.load(f)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Find any test image
sample = next(TEST_DIR.rglob('*.jpg'))
print(f"Using sample image: {sample.name}")
img = tfms(Image.open(sample).convert('RGB')).unsqueeze(0).to(device)

# Warm-up runs (first inference is always slower)
print("Warming up...")
with torch.no_grad():
    for _ in range(3):
        _ = model(img)

# Actual benchmark
N = 30
print(f"Running {N} inference iterations...")
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        _ = model(img)
elapsed_ms = (time.time() - t0) / N * 1000

print(f"\n{'=' * 50}")
print(f"Average inference time: {elapsed_ms:.1f} ms per image")
print(f"{'=' * 50}")
print(f"\nFor the 3-second KPI, your budget breakdown is:")
print(f"  Model inference: {elapsed_ms:.0f} ms")
print(f"  Image preprocessing: ~50 ms")
print(f"  HTTP round trip: ~200-500 ms (varies)")
print(f"  Headroom: {3000 - elapsed_ms - 50 - 500:.0f} ms")
if elapsed_ms < 500:
    print("  Status: ✓ Well within budget")
elif elapsed_ms < 1500:
    print("  Status: ⚠ Tight but workable")
else:
    print("  Status: ✗ Won't hit KPI — consider quantizing the model")
