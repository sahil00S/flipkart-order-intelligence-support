from pathlib import Path
import json

import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn


MODEL_PATH = Path("models/product_classifier.pt")

CLASS_NAMES = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        10,
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


def classify_product_image(image_path: str) -> dict:
    """
    Classify a real committed Fashion-MNIST test image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = load_model()

    transform = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    image = Image.open(
        image_path
    ).convert("L")

    tensor = transform(image).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    category = CLASS_NAMES[
        prediction.item()
    ]

    return {
        "category": category,
        "confidence": float(
            confidence.item()
        ),
    }


if __name__ == "__main__":

    sample_image = (
        "data/sample_images/"
        "test_0000_true_9_Ankle_boot.png"
    )

    result = classify_product_image(
        sample_image
    )

    print(
        "=== Image Classification Tool Test ==="
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )