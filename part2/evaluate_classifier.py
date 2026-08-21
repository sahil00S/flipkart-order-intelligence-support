import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

from PIL import Image


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 128

DATA_DIR = "data/fashion_mnist"
MODEL_PATH = "models/product_classifier.pt"

CONFUSION_MATRIX_PATH = "part2/confusion_matrix.npy"
METRICS_PATH = "part2/per_class_metrics.csv"

SAMPLE_DIR = "data/sample_images"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Fashion-MNIST class names
# ============================================================

class_names = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot"
]


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Test Dataset
# ============================================================

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print("Test samples:", len(test_dataset))


# ============================================================
# Load ResNet-18 architecture
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Generate Predictions
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        ).cpu().numpy()

        all_predictions.extend(
            predictions
        )

        all_labels.extend(
            labels.numpy()
        )


all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# ============================================================
# Test Accuracy
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print()
print("=== Test Accuracy ===")
print(
    "Accuracy:",
    round(test_accuracy, 4)
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=np.arange(10)
)

print()
print("=== 10x10 Confusion Matrix ===")
print(cm)

np.save(
    CONFUSION_MATRIX_PATH,
    cm
)


# ============================================================
# Per-Class Precision / Recall
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=np.arange(10),
        zero_division=0
    )
)

metrics_rows = []

print()
print("=== Per-Class Precision / Recall ===")

for i in range(10):

    print(
        class_names[i],
        "| Precision:",
        round(precision[i], 4),
        "| Recall:",
        round(recall[i], 4),
        "| F1:",
        round(f1[i], 4),
        "| Support:",
        support[i]
    )

    metrics_rows.append({
        "class_id": i,
        "class_name": class_names[i],
        "precision": precision[i],
        "recall": recall[i],
        "f1": f1[i],
        "support": support[i]
    })


import pandas as pd

metrics_df = pd.DataFrame(
    metrics_rows
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# Real Confusion Pairs
# ============================================================

print()
print("=== Strongest Confusion Pairs ===")

pair_values = []

for true_class in range(10):

    for predicted_class in range(10):

        if true_class == predicted_class:
            continue

        pair_values.append({
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": cm[
                true_class,
                predicted_class
            ]
        })


pair_values = sorted(
    pair_values,
    key=lambda x: x["count"],
    reverse=True
)

for pair in pair_values[:10]:

    print(
        class_names[pair["true_class"]],
        "->",
        class_names[pair["predicted_class"]],
        ":",
        pair["count"]
    )


# ============================================================
# Export At Least 5 Real Test Images
# ============================================================

os.makedirs(
    SAMPLE_DIR,
    exist_ok=True
)

print()
print("=== Exporting Sample Images ===")

# Select the first five test images
# These are REAL Fashion-MNIST test images.
sample_indices = [0, 1, 2, 3, 4]

for index in sample_indices:

    image, true_label = test_dataset[index]

    # Load original untransformed Fashion-MNIST image
    original_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=None
    )

    original_image, original_label = (
        original_dataset[index]
    )

    filename = (
        f"test_{index:04d}"
        f"_true_{original_label}"
        f"_{class_names[original_label]}.png"
    )

    path = os.path.join(
        SAMPLE_DIR,
        filename
    )

    original_image.save(path)

    print(
        "Saved:",
        path
    )


print()
print("=== Evaluation Complete ===")
print("Confusion matrix:", CONFUSION_MATRIX_PATH)
print("Per-class metrics:", METRICS_PATH)
print("Sample images:", SAMPLE_DIR)

# ============================================================
# Save Human-Readable Confusion Matrix
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("Fashion-MNIST Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    np.arange(10),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(10),
    class_names
)

for i in range(10):
    for j in range(10):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "part2/confusion_matrix.png",
    dpi=200
)

plt.close()

print(
    "Confusion matrix image saved to: "
    "part2/confusion_matrix.png"
)