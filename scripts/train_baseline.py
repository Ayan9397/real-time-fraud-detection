from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TARGET = "isFraud"


# Keep the first baseline deliberately small.
BASELINE_FEATURES = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
]


def load_data(filename):
    return pd.read_csv(
        DATA_DIR / filename,
        usecols=BASELINE_FEATURES + [TARGET],
    )


def build_pipeline(X):

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(model, X, y, name):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    print("\n" + "=" * 70)
    print(f"{name.upper()} RESULTS")
    print("=" * 70)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():

    print("=" * 70)
    print("LOADING BASELINE DATA")
    print("=" * 70)

    train = load_data("train.csv")
    valid = load_data("valid.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_valid = valid.drop(columns=[TARGET])
    y_valid = valid[TARGET]

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_valid.shape}")

    print("\nFeatures used:")
    for feature in BASELINE_FEATURES:
        print(f"  {feature}")

    print("\nBuilding pipeline...")

    pipeline = build_pipeline(X_train)

    print("Fitting Logistic Regression...")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    evaluate(
        pipeline,
        X_train,
        y_train,
        "train",
    )

    evaluate(
        pipeline,
        X_valid,
        y_valid,
        "validation",
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR / "logistic_baseline.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(f"Path: {model_path}")


if __name__ == "__main__":
    main()