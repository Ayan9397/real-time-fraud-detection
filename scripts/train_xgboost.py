from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OrdinalEncoder


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


TARGET = "isFraud"


def get_features(df):

    features = [
        column
        for column in df.columns
        if column != TARGET
        and column != "TransactionID"
    ]

    return features


def load_dataset(filename):

    file_path = DATA_DIR / filename

    df = pd.read_csv(file_path)

    return df


def build_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print(
        f"Numeric features: {len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
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
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
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

    return preprocessor


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

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

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
    print("XGBOOST FRAUD MODEL")
    print("=" * 70)

    print("\nLoading training data...")

    train = load_dataset("train.csv")

    print(
        f"Train shape: {train.shape}"
    )

    print("\nLoading validation data...")

    valid = load_dataset("valid.csv")

    print(
        f"Validation shape: {valid.shape}"
    )

    X_train = train.drop(
        columns=[TARGET, "TransactionID"]
    )

    y_train = train[TARGET]

    X_valid = valid.drop(
        columns=[TARGET, "TransactionID"]
    )

    y_valid = valid[TARGET]

    print("\nPreparing preprocessing pipeline...")

    preprocessor = build_preprocessor(
        X_train
    )

    print("\nFitting preprocessing...")

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        "Training data transformed."
    )

    print(
        f"Transformed shape: "
        f"{X_train_transformed.shape}"
    )

    print("\nTransforming validation data...")

    X_valid_transformed = (
        preprocessor.transform(
            X_valid
        )
    )

    print(
        "Validation data transformed."
    )

    fraud_count = y_train.sum()

    legitimate_count = (
        len(y_train) - fraud_count
    )

    scale_pos_weight = (
        legitimate_count / fraud_count
    )

    print("\nClass balance:")

    print(
        f"Legitimate: "
        f"{legitimate_count:,}"
    )

    print(
        f"Fraud: "
        f"{fraud_count:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    print("\nCreating XGBoost model...")

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[
            (
                X_valid_transformed,
                y_valid,
            )
        ],
        verbose=50,
    )

    print("\nTraining completed.")

    evaluate(
        model,
        X_train_transformed,
        y_train,
        "train",
    )

    evaluate(
        model,
        X_valid_transformed,
        y_valid,
        "validation",
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "xgboost_fraud_model.joblib"
    )

    preprocessor_path = (
        MODEL_DIR
        / "xgboost_preprocessor.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print("\n" + "=" * 70)
    print("MODEL ARTIFACTS SAVED")
    print("=" * 70)

    print(
        f"Model: {model_path}"
    )

    print(
        f"Preprocessor: "
        f"{preprocessor_path}"
    )


if __name__ == "__main__":
    main()