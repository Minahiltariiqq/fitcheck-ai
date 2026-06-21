"""
ml/scripts/03_evaluate.py

Runs the trained model against the test set and prints
precision/recall/F1 per class. Saves a confusion matrix image.

Run AFTER you've trained the model (in Colab) and copied
fitcheck_mobilenet.pth + labels.json into ml/models/.

Usage:
    python 03_evaluate.py
"""
import torch
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data' / 'processed'
MODEL_PATH = SCRIPT_DIR.parent / 'models' / 'fitcheck_mobilenet.pth'
LABELS_PATH = SCRIPT_DIR.parent / 'models' / 'labels.json'
REPORT_DIR = SCRIPT_DIR.parent / 'reports'
REPORT_DIR.mkdir(exist_ok=True)

# Dell laptop has no GPU, so use CPU. This script is slow but only runs once.
device = torch.device('cpu')
print(f"Using device: {device}")

if not MODEL_PATH.exists():
    print(f"ERROR: Model file missing: {MODEL_PATH}")
    print("Train the model in Google Colab first, download the .pth file,")
    print("and put it in ml/models/")
    exit(1)

with open(LABELS_PATH) as f:
    classes = json.load(f)

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(DATA_DIR / 'test', transform=tfms)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"Test images: {len(test_ds)}")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

print("Running inference on test set...")
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        out = model(imgs.to(device))
        all_preds += out.argmax(1).cpu().tolist()
        all_labels += labels.tolist()

# Text report
report = classification_report(all_labels, all_preds, target_names=classes)
print("\n" + report)

# Save to file for your FYP report screenshots
with open(REPORT_DIR / 'classification_report.txt', 'w') as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — FitCheck AI')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORT_DIR / 'confusion_matrix.png', dpi=150)
print(f"\nSaved confusion matrix to {REPORT_DIR / 'confusion_matrix.png'}")
print(f"Saved text report to {REPORT_DIR / 'classification_report.txt'}")
