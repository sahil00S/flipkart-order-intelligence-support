from pathlib import Path
import json

import joblib
import pandas as pd


MODEL_PATH = Path("models/return_risk_model.pkl")
THRESHOLD_PATH = Path("models/return_risk_threshold.json")


def check_return_risk(order_features: dict) -> dict:
    """
    Predict return risk using the actual saved Part 1 model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        threshold_data = json.load(f)

    threshold = float(
        threshold_data["t_star_rf"]
    )

    # The saved pipeline contains the preprocessing,
    # so the tool sends raw order features directly.
    features = pd.DataFrame(
        [order_features]
    )

    probabilities = model.predict_proba(features)

    return_probability = float(
        probabilities[0, 1]
    )

    if return_probability >= threshold:
        risk_bucket = "HIGH"
    else:
        risk_bucket = "LOW"

    return {
        "return_probability": return_probability,
        "threshold": threshold,
        "risk_bucket": risk_bucket,
    }


if __name__ == "__main__":

    # Realistic order example using the exact
    # feature names expected by the Part 1 model.
    example_order = {
        "product_category": "Apparel",
        "price_inr": 1200,
        "discount_pct": 25,
        "payment_method": "COD",
        "customer_tenure_days": 300,
        "num_previous_orders": 8,
        "num_previous_returns": 2,
        "delivery_distance_km": 150,
        "delivery_days": 5,
        "is_weekend_order": 0,
        "rating_given": 3,
    }

    result = check_return_risk(
        example_order
    )

    print("=== Return Risk Tool Test ===")
    print(json.dumps(result, indent=2))