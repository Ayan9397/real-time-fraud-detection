from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBClassifier


TRAIN_FILE = Path("data/processed/train.csv")
VALID_FILE = Path("data/processed/valid.csv")
MODEL_DIR = Path("models")

TARGET = "isFraud"
ID_COLUMN = "TransactionID"


def evaluate_model(model, X, y, dataset_name):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))


def main():

    print("=" * 70)
    print("MISSINGNESS-AWARE XGBOOST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)

    print("Train shape:", train_df.shape)
    print("Validation shape:", valid_df.shape)

    # ---------------------------------------------------------
    # Separate target
    # ---------------------------------------------------------

    y_train = train_df[TARGET]
    y_valid = valid_df[TARGET]

    X_train = train_df.drop(
        columns=[TARGET, ID_COLUMN]
    ).copy()

    X_valid = valid_df.drop(
        columns=[TARGET, ID_COLUMN]
    ).copy()

    print("\nTarget distribution:")
    print(y_train.value_counts())

    # ---------------------------------------------------------
    # Identify original feature types
    # ---------------------------------------------------------

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print("\nOriginal numeric features:", len(numeric_features))
    print(
        "Original categorical features:",
        len(categorical_features),
    )

    # ---------------------------------------------------------
    # Missingness indicators
    #
    # Only create indicators for original features.
    # This avoids accidentally treating the indicator columns
    # as categorical variables.
    # ---------------------------------------------------------

    print("\nCreating missingness indicators...")

    missing_train = X_train.isna().astype("int8")
    missing_valid = X_valid.isna().astype("int8")

    missing_train.columns = [
        f"{column}_missing"
        for column in missing_train.columns
    ]

    missing_valid.columns = [
        f"{column}_missing"
        for column in missing_valid.columns
    ]

    X_train = pd.concat(
        [X_train, missing_train],
        axis=1,
    )

    X_valid = pd.concat(
        [X_valid, missing_valid],
        axis=1,
    )

    # Missing indicators are all numeric.
    missing_features = missing_train.columns.tolist()

    numeric_features_extended = (
        numeric_features + missing_features
    )

    print(
        "Features after indicators:",
        X_train.shape[1],
    )

    print(
        "Numeric features after indicators:",
        len(numeric_features_extended),
    )

    print(
        "Categorical features:",
        len(categorical_features),
    )

    # ---------------------------------------------------------
    # Numeric preprocessing
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Categorical preprocessing
    # ---------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
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

    # ---------------------------------------------------------
    # Combined preprocessing
    # ---------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features_extended,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    print("\nFitting preprocessing pipeline...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_valid_processed = preprocessor.transform(
        X_valid
    )

    print(
        "Processed train shape:",
        X_train_processed.shape,
    )

    print(
        "Processed validation shape:",
        X_valid_processed.shape,
    )

    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------

    print("\nChecking processed data...")

    print(
        "Processed data type:",
        X_train_processed.dtype,
    )

    if not pd.api.types.is_numeric_dtype(
        X_train_processed.dtype
    ):
        raise TypeError(
            "Processed matrix is not numeric."
        )

    print("Numeric matrix check: PASSED")

    # ---------------------------------------------------------
    # Class imbalance
    # ---------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ---------------------------------------------------------
    # XGBoost
    #
    # Same configuration as our previous model so that
    # the experiment isolates the effect of missingness
    # indicators.
    # ---------------------------------------------------------

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
    print("This may take some time.")

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[
            (
                X_valid_processed,
                y_valid,
            )
        ],
        verbose=50,
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    evaluate_model(
        model,
        X_train_processed,
        y_train,
        "TRAINING RESULTS",
    )

    evaluate_model(
        model,
        X_valid_processed,
        y_valid,
        "VALIDATION RESULTS",
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "xgboost_missing_model.joblib"
    )

    preprocessor_path = (
        MODEL_DIR
        / "xgboost_missing_preprocessor.joblib"
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
    print("MODEL SAVED")
    print("=" * 70)

    print(f"Model:        {model_path}")
    print(f"Preprocessor: {preprocessor_path}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()