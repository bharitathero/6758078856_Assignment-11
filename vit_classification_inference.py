import os

import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification


DATA_DIR = "/opt/project/dataset/pizza_not_pizza"
MODEL_PATH = "/opt/project/tmp/class11_vit/vit_model_best.pth"
SPLIT = "val"
BATCH_SIZE = 8
NUM_WORKERS = 0
MODEL_NAME = "google/vit-base-patch16-224-in21k"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

split_dir = os.path.join(DATA_DIR, SPLIT)
if not os.path.isdir(split_dir):
    raise FileNotFoundError(f"Missing evaluation split folder: {split_dir}")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    class_names = checkpoint.get("class_names")
    model_name = checkpoint.get("model_name", MODEL_NAME)
    image_size = checkpoint.get("image_size", 224)
    image_mean = checkpoint.get("image_mean", [0.485, 0.456, 0.406])
    image_std = checkpoint.get("image_std", [0.229, 0.224, 0.225])
    best_val_metrics = checkpoint.get("best_val_metrics")
    best_threshold = checkpoint.get("best_val_metrics", {}).get("threshold", 0.5)
else:
    state_dict = checkpoint
    class_names = None
    model_name = MODEL_NAME
    image_size = 224
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    best_val_metrics = None
    best_threshold = 0.5

resize_size = int(image_size * 256 / 224)
data_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(image_mean, image_std),
])

dataset = datasets.ImageFolder(split_dir, data_transform)
if class_names is None:
    class_names = dataset.classes
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(class_names),
    ignore_mismatched_sizes=True,
).to(device)

model.load_state_dict(state_dict)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=inputs).logits
        if outputs.shape[1] == 2:
            positive_probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (positive_probs >= best_threshold).long()
        else:
            preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("=" * 60)
print("Validation Results")
print("=" * 60)
print(f"Device: {device}")
print(f"Data directory: {DATA_DIR}")
print(f"Model checkpoint: {MODEL_PATH}")
if best_val_metrics is not None:
    print(
        f"Saved Best Validation Metrics: "
        f"Epoch {best_val_metrics['epoch']} ({best_val_metrics['stage']}) | "
        f"Loss: {best_val_metrics['loss']:.4f} | "
        f"Acc: {best_val_metrics['accuracy']:.4f} | "
        f"F1: {best_val_metrics['f1']:.4f} | "
        f"Threshold: {best_threshold:.3f}"
    )
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

F1_score = f1
print(f"\nF1_score = {F1_score:.4f}")
