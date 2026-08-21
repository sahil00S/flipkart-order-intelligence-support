import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 128
NUM_WORKERS = 0

NUM_CLASSES = 10
VAL_SIZE = 5000

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 3

HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-4

DATA_DIR = "data/fashion_mnist"
MODEL_PATH = "models/product_classifier.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# ImageNet normalization
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Fashion-MNIST transforms
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# Load Fashion-MNIST
# ============================================================

train_full = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

print()
print("=== Dataset Sizes ===")
print("Training:", len(train_full))
print("Test:", len(test_dataset))


# ============================================================
# Stratified 55k / 5k split
# ============================================================

targets = np.asarray(train_full.targets)

indices = np.arange(len(train_full))

train_indices, val_indices = train_test_split(
    indices,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=targets
)

train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    train_full,
    val_indices
)

print("Training subset:", len(train_dataset))
print("Validation subset:", len(val_dataset))


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# Load pretrained ResNet-18
# ============================================================

print()
print("=== Loading Pretrained ResNet-18 ===")

weights = models.ResNet18_Weights.DEFAULT

resnet = models.resnet18(
    weights=weights
)

resnet = resnet.to(DEVICE)


# ============================================================
# Feature Extractor
# ============================================================

# Remove the original ImageNet classifier.
# ResNet-18 produces 512-dimensional features.

feature_extractor = nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(DEVICE)

feature_extractor.eval()

for parameter in feature_extractor.parameters():
    parameter.requires_grad = False


# ============================================================
# Feature Extraction Function
# ============================================================

@torch.no_grad()
def extract_features(loader):

    all_features = []
    all_labels = []

    for batch_number, (images, labels) in enumerate(loader):

        images = images.to(DEVICE)

        features = feature_extractor(images)

        # [batch, 512, 1, 1] -> [batch, 512]
        features = features.flatten(1)

        all_features.append(
            features.cpu()
        )

        all_labels.append(
            labels
        )

        if (batch_number + 1) % 50 == 0:
            print(
                "Processed batches:",
                batch_number + 1
            )

    return (
        torch.cat(all_features),
        torch.cat(all_labels)
    )


# ============================================================
# Extract Frozen Features
# ============================================================

print()
print("=== Extracting Training Features ===")

train_features, train_labels = extract_features(
    train_loader
)

print(
    "Training feature shape:",
    tuple(train_features.shape)
)

print()
print("=== Extracting Validation Features ===")

val_features, val_labels = extract_features(
    val_loader
)

print(
    "Validation feature shape:",
    tuple(val_features.shape)
)


# ============================================================
# Feature DataLoaders
# ============================================================

feature_train_loader = DataLoader(
    TensorDataset(
        train_features,
        train_labels
    ),
    batch_size=256,
    shuffle=True
)

feature_val_loader = DataLoader(
    TensorDataset(
        val_features,
        val_labels
    ),
    batch_size=256,
    shuffle=False
)


# ============================================================
# New 10-Class Classifier
# ============================================================

classifier = nn.Linear(
    512,
    NUM_CLASSES
)

classifier = classifier.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=HEAD_LR
)


# ============================================================
# Classifier Training
# ============================================================

def train_classifier_epoch():

    classifier.train()

    correct = 0
    total = 0
    total_loss = 0.0

    for features, labels in feature_train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = classifier(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


@torch.no_grad()
def evaluate_classifier():

    classifier.eval()

    correct = 0
    total = 0

    for features, labels in feature_val_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = classifier(features)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return correct / total


# ============================================================
# Train Feature-Extraction Classifier
# ============================================================

print()
print("=== Training Classifier Head ===")

best_val_accuracy = 0.0
best_classifier_state = None

for epoch in range(HEAD_EPOCHS):

    train_loss, train_accuracy = train_classifier_epoch()

    val_accuracy = evaluate_classifier()

    print(
        f"Head Epoch {epoch + 1}/{HEAD_EPOCHS} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_classifier_state = {
            key: value.cpu().clone()
            for key, value in classifier.state_dict().items()
        }


if best_classifier_state is not None:

    classifier.load_state_dict(
        best_classifier_state
    )


print()
print(
    "Best feature-extraction validation accuracy:",
    round(best_val_accuracy, 4)
)


# ============================================================
# Conditional Fine-Tuning
# ============================================================

if best_val_accuracy < 0.80:

    print()
    print("Validation accuracy < 80%.")
    print("Starting later-layer fine-tuning.")

    # Rebuild full ResNet with 10-class classifier
    model = models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    # Load our trained classifier into ResNet
    model.fc.load_state_dict(
        classifier.state_dict()
    )

    model = model.to(DEVICE)

    # Freeze everything
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze later layer + classifier
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer_fine = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=FINE_TUNE_LR
    )

    best_fine_accuracy = best_val_accuracy
    best_fine_state = None

    for epoch in range(FINE_TUNE_EPOCHS):

        model.train()

        correct = 0
        total = 0
        total_loss = 0.0

        for images, labels in DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer_fine.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer_fine.step()

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_accuracy = correct / total

        # Validation
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(
            f"Fine Epoch {epoch + 1}/{FINE_TUNE_EPOCHS} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_fine_accuracy:

            best_fine_accuracy = val_accuracy

            best_fine_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_fine_state is not None:

        model.load_state_dict(
            best_fine_state
        )

    best_val_accuracy = best_fine_accuracy

else:

    print()
    print("Validation accuracy >= 80%.")
    print("Fine-tuning not required.")

    # Build final model from feature extractor + classifier
    model = models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model.fc.load_state_dict(
        classifier.state_dict()
    )

    model = model.to(DEVICE)


# ============================================================
# Final Test Accuracy
# ============================================================

print()
print("=== Final Test Evaluation ===")

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)

test_accuracy = test_correct / test_total

print(
    "Validation accuracy:",
    round(best_val_accuracy, 4)
)

print(
    "Test accuracy:",
    round(test_accuracy, 4)
)


# ============================================================
# Save Model
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print(
    "Model saved to:",
    MODEL_PATH
)
