import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance


df = pd.read_csv("orders_dataset.csv")

print("Dataset shape:", df.shape)
print()
print("Missing values:")
print(df["rating_given"].isna().sum())

overall_missing_pct = df["rating_given"].isna().mean() * 100

cod_missing_pct = (
    df.loc[df["payment_method"] == "COD", "rating_given"]
    .isna()
    .mean()
    * 100
)

non_cod_missing_pct = (
    df.loc[df["payment_method"] != "COD", "rating_given"]
    .isna()
    .mean()
    * 100
)

print()
print("Overall missing rating:", round(overall_missing_pct, 2), "%")
print("COD missing rating:", round(cod_missing_pct, 2), "%")
print("Non-COD missing rating:", round(non_cod_missing_pct, 2), "%")

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns=["returned", "order_id"])
y = df["returned"]

# Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

print()
print("Training target distribution:")
print(y_train.value_counts(normalize=True))

print()
print("Test target distribution:")
print(y_test.value_counts(normalize=True))

numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]

categorical_features = [
    "product_category",
    "payment_method",
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# Fit preprocessing only on training data
X_train_transformed = preprocessor.fit_transform(X_train)

print()
print("Original training features:", X_train.shape)
print("Transformed training features:", X_train_transformed.shape)
print("Missing values after preprocessing:",
      np.isnan(X_train_transformed.toarray()).sum()
      if hasattr(X_train_transformed, "toarray")
      else np.isnan(X_train_transformed).sum())

# ============================================================
# DummyClassifier Baseline
# ============================================================

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Transform test data using the already-fitted preprocessor
X_test_transformed = preprocessor.transform(X_test)

# Dummy baseline: always predicts the majority class
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train_transformed, y_train)

dummy_predictions = dummy_model.predict(X_test_transformed)

dummy_accuracy = accuracy_score(y_test, dummy_predictions)
dummy_precision = precision_score(
    y_test, dummy_predictions, zero_division=0
)
dummy_recall = recall_score(
    y_test, dummy_predictions, zero_division=0
)
dummy_f1 = f1_score(
    y_test, dummy_predictions, zero_division=0
)

print()
print("=== DummyClassifier Baseline ===")
print("Accuracy:", round(dummy_accuracy, 4))
print("Precision:", round(dummy_precision, 4))
print("Recall:", round(dummy_recall, 4))
print("F1:", round(dummy_f1, 4))
print("F1:", round(dummy_f1, 4))

# ============================================================
# Logistic Regression
# ============================================================

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_transformed, y_train)

# Probability of returned = 1
logistic_probabilities = logistic_model.predict_proba(
    X_test_transformed
)[:, 1]


# ============================================================
# Default Threshold = 0.50
# ============================================================

logistic_default_predictions = (
    logistic_probabilities >= 0.50
).astype(int)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_default_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_default_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_default_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_default_predictions,
    zero_division=0
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

print()
print("=== Logistic Regression (Threshold = 0.50) ===")
print("Accuracy:", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall:", round(logistic_recall, 4))
print("F1:", round(logistic_f1, 4))
print("ROC-AUC:", round(logistic_roc_auc, 4))


# ============================================================
# Threshold Sweep: 0.10 to 0.90
# Step = 0.01
# ============================================================

thresholds = np.arange(0.10, 0.901, 0.01)

threshold_results = []

for threshold in thresholds:

    predictions = (
        logistic_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# Convert results to a table
threshold_df = pd.DataFrame(threshold_results)

print()
print("=== Logistic Regression Threshold Sweep ===")
print(threshold_df.to_string(index=False))


# ============================================================
# F1-Maximising Threshold
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])
best_precision = float(best_row["precision"])
best_recall = float(best_row["recall"])
best_f1 = float(best_row["f1"])

print()
print("=== F1-Maximising Threshold ===")
print("Best threshold:", best_threshold)
print("Precision:", round(best_precision, 4))
print("Recall:", round(best_recall, 4))
print("F1:", round(best_f1, 4))

# ============================================================
# Random Forest + GridSearchCV
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Build the complete leakage-safe pipeline
# Preprocessing is fitted separately inside every CV fold.
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),
    ]
)

# Required parameter grid
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10, None],
}

# Required 5-fold StratifiedKFold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Grid search using ROC-AUC
rf_grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
    return_train_score=False
)

print()
print("=== Random Forest GridSearchCV ===")
print("Starting 5-fold cross-validation...")

rf_grid_search.fit(X_train, y_train)

# Winning fitted pipeline
best_rf_pipeline = rf_grid_search.best_estimator_

# Best CV result
best_rf_cv_auc = rf_grid_search.best_score_

# Held-out test predictions
rf_test_probabilities = best_rf_pipeline.predict_proba(
    X_test
)[:, 1]

rf_test_auc = roc_auc_score(
    y_test,
    rf_test_probabilities
)

print()
print("=== Random Forest Results ===")
print("Best parameters:", rf_grid_search.best_params_)
print("Best CV ROC-AUC:", round(best_rf_cv_auc, 4))
print("Test ROC-AUC:", round(rf_test_auc, 4))
print(
    "CV/Test ROC-AUC difference:",
    round(abs(best_rf_cv_auc - rf_test_auc), 4)
)

# ============================================================
# Feature Importance
# ============================================================

print()
print("=== Feature Importance ===")

# Get the fitted preprocessing transformer
fitted_preprocessor = best_rf_pipeline.named_steps["preprocessor"]

# Get the fitted Random Forest
fitted_rf = best_rf_pipeline.named_steps["model"]

# Get feature names after preprocessing
feature_names = fitted_preprocessor.get_feature_names_out()

# Impurity-based feature importance
impurity_importance = fitted_rf.feature_importances_

impurity_df = pd.DataFrame({
    "feature": feature_names,
    "impurity_importance": impurity_importance
})

impurity_df = impurity_df.sort_values(
    "impurity_importance",
    ascending=False
)

print()
print("=== Top 5 Impurity-Based Features ===")
print(
    impurity_df.head(5).to_string(index=False)
)


# ============================================================
# Permutation Importance on Held-Out Test Set
# ============================================================

print()
print("=== Permutation Importance ===")
print("Calculating on held-out test set...")

permutation_result = permutation_importance(
    best_rf_pipeline,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# IMPORTANT:
# permutation_importance is being run on the complete pipeline,
# so the feature names here correspond to the ORIGINAL input
# features, not the one-hot encoded columns.

permutation_df = pd.DataFrame({
    "feature": X_test.columns,
    "permutation_importance_mean": permutation_result.importances_mean,
    "permutation_importance_std": permutation_result.importances_std
})

permutation_df = permutation_df.sort_values(
    "permutation_importance_mean",
    ascending=False
)

print()
print("=== Top 5 Permutation Features ===")
print(
    permutation_df.head(5).to_string(index=False)
)

# ============================================================
# Subgroup Analysis
# ============================================================

print()
print("=== Subgroup Analysis ===")

# Use the winning Random Forest at the default 0.50 threshold
rf_default_predictions = (
    rf_test_probabilities >= 0.50
).astype(int)


# ------------------------------------------------------------
# Product Category
# ------------------------------------------------------------

product_category_results = []

for category in sorted(X_test["product_category"].unique()):

    mask = X_test["product_category"] == category

    category_precision = precision_score(
        y_test[mask],
        rf_default_predictions[mask],
        zero_division=0
    )

    category_recall = recall_score(
        y_test[mask],
        rf_default_predictions[mask],
        zero_division=0
    )

    category_count = mask.sum()

    product_category_results.append({
        "product_category": category,
        "n": category_count,
        "precision": category_precision,
        "recall": category_recall
    })

product_category_df = pd.DataFrame(
    product_category_results
)

print()
print("=== Precision / Recall by Product Category ===")
print(
    product_category_df.to_string(index=False)
)


# ------------------------------------------------------------
# Payment Method
# ------------------------------------------------------------

payment_method_results = []

for payment_method in sorted(X_test["payment_method"].unique()):

    mask = X_test["payment_method"] == payment_method

    payment_precision = precision_score(
        y_test[mask],
        rf_default_predictions[mask],
        zero_division=0
    )

    payment_recall = recall_score(
        y_test[mask],
        rf_default_predictions[mask],
        zero_division=0
    )

    payment_count = mask.sum()

    payment_method_results.append({
        "payment_method": payment_method,
        "n": payment_count,
        "precision": payment_precision,
        "recall": payment_recall
    })

payment_method_df = pd.DataFrame(
    payment_method_results
)

print()
print("=== Precision / Recall by Payment Method ===")
print(
    payment_method_df.to_string(index=False)
)


import json
import joblib

# ============================================================
# Final Random Forest Threshold (t*_rf)
# ============================================================

print()
print("=== Final Random Forest Threshold Sweep ===")

# IMPORTANT:
# These probabilities come from the FINAL winning Random Forest
# pipeline selected by GridSearchCV.
rf_thresholds = np.arange(0.10, 0.901, 0.01)

rf_threshold_results = []

for threshold in rf_thresholds:

    predictions = (
        rf_test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    rf_threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


rf_threshold_df = pd.DataFrame(
    rf_threshold_results
)

best_rf_threshold_row = rf_threshold_df.loc[
    rf_threshold_df["f1"].idxmax()
]

t_star_rf = float(
    best_rf_threshold_row["threshold"]
)

t_star_rf_precision = float(
    best_rf_threshold_row["precision"]
)

t_star_rf_recall = float(
    best_rf_threshold_row["recall"]
)

t_star_rf_f1 = float(
    best_rf_threshold_row["f1"]
)

print()
print("=== t*_rf ===")
print("Best RF threshold:", t_star_rf)
print("Precision:", round(t_star_rf_precision, 4))
print("Recall:", round(t_star_rf_recall, 4))
print("F1:", round(t_star_rf_f1, 4))


# ============================================================
# Final Model Save
# ============================================================

model_path = "models/return_risk_model.pkl"

joblib.dump(
    best_rf_pipeline,
    model_path
)

print()
print("=== Final Model Saved ===")
print("Path:", model_path)


# ============================================================
# Save Threshold Information for Part 3
# ============================================================

threshold_info = {
    "t_star_rf": t_star_rf,
    "precision_at_t_star_rf": t_star_rf_precision,
    "recall_at_t_star_rf": t_star_rf_recall,
    "f1_at_t_star_rf": t_star_rf_f1,
    "model_type": "RandomForestClassifier",
    "best_parameters": rf_grid_search.best_params_,
    "test_roc_auc": float(rf_test_auc),
    "cv_roc_auc": float(best_rf_cv_auc)
}

with open(
    "models/return_risk_threshold.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        threshold_info,
        f,
        indent=2
    )

print("Threshold information saved:")
print("models/return_risk_threshold.json")

# ============================================================
# Subgroup Finding
# ============================================================

weakest_payment_subgroup = payment_method_df.loc[
    payment_method_df["recall"].idxmin()
]

print()
print("=== Weakest Payment-Method Subgroup ===")
print(
    "Subgroup:",
    weakest_payment_subgroup["payment_method"]
)
print(
    "Precision:",
    round(weakest_payment_subgroup["precision"], 4)
)
print(
    "Recall:",
    round(weakest_payment_subgroup["recall"], 4)
)
print(
    "Proposed fix: use a payment-method-specific decision "
    "threshold for this subgroup to improve recall while "
    "monitoring the resulting precision trade-off."
)

