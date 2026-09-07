from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("models/xgboost_missing_model.joblib")
PREPROCESSOR_PATH = Path("models/xgboost_missing_preprocessor.joblib")
VALID_PATH = Path("data/processed/valid.csv")
OUTPUT_DIR = Path("models/shap")

SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def add_missing_indicators(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Add one binary missingness indicator for every original feature.

    IMPORTANT:
    The model was trained using the naming convention:
        feature_missing

    Therefore we must use a SINGLE underscore.
    """

    df = df.copy()

    indicators = pd.DataFrame(
        {
            f"{column}_missing": df[column].isna().astype("int8")
            for column in feature_columns
        },
        index=df.index,
    )

    return pd.concat([df, indicators], axis=1)


def main() -> None:

    print("Loading model and preprocessor...")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Loading validation data...")

    df = pd.read_csv(VALID_PATH)

    target = "isFraud"
    id_column = "TransactionID"

    feature_columns = [
        column
        for column in df.columns
        if column not in {target, id_column}
    ]

    print(f"Validation shape: {df.shape}")
    print(f"Original feature count: {len(feature_columns)}")

    # ---------------------------------------------------------
    # 1. Create the same features used during model training
    # ---------------------------------------------------------

    X = df[feature_columns].copy()

    X = add_missing_indicators(
        X,
        feature_columns,
    )

    print(
        f"Feature count after missing indicators: {X.shape[1]}"
    )

    # ---------------------------------------------------------
    # 2. Match the exact schema used by the saved preprocessor
    # ---------------------------------------------------------

    expected_columns = list(
        preprocessor.feature_names_in_
    )

    print(
        f"Preprocessor expects: {len(expected_columns)} columns"
    )

    missing_columns = set(expected_columns) - set(X.columns)

    if missing_columns:
        raise ValueError(
            "Required columns are missing from SHAP input: "
            f"{missing_columns}"
        )

    extra_columns = set(X.columns) - set(expected_columns)

    if extra_columns:
        raise ValueError(
            "Unexpected columns found in SHAP input: "
            f"{extra_columns}"
        )

    # Reorder exactly according to the fitted preprocessor.
    X = X[expected_columns]

    print(
        "Feature schema matched to saved preprocessor."
    )

    # ---------------------------------------------------------
    # 3. Sample validation transactions
    # ---------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(X),
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        len(X),
        size=sample_size,
        replace=False,
    )

    X_sample = X.iloc[sample_indices].copy()

    print(
        f"SHAP sample size: {len(X_sample)}"
    )

    # ---------------------------------------------------------
    # 4. Apply the exact saved preprocessing pipeline
    # ---------------------------------------------------------

    print("Applying preprocessing...")

    X_processed = preprocessor.transform(
        X_sample
    )

    # Convert sparse matrix to dense if necessary.
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    X_processed = np.asarray(
        X_processed
    )

    print(
        f"Processed SHAP shape: {X_processed.shape}"
    )

    # ---------------------------------------------------------
    # 5. Get final processed feature names
    # ---------------------------------------------------------

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    print(
        f"Processed feature names: {len(feature_names)}"
    )

    if X_processed.shape[1] != len(feature_names):
        raise ValueError(
            "Feature count mismatch between processed "
            "data and feature names."
        )

    # ---------------------------------------------------------
    # 6. Create SHAP TreeExplainer
    # ---------------------------------------------------------

    print("Creating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(
        model
    )

    # ---------------------------------------------------------
    # 7. Calculate SHAP values
    # ---------------------------------------------------------

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_processed
    )

    # Binary XGBoost compatibility.
    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    print(
        f"SHAP value shape: {shap_values.shape}"
    )

    # ---------------------------------------------------------
    # 8. Validate SHAP dimensions
    # ---------------------------------------------------------

    if shap_values.shape != X_processed.shape:
        raise ValueError(
            "SHAP values shape does not match "
            "processed feature matrix."
        )

    print(
        "SHAP dimensions verified successfully."
    )

    # ---------------------------------------------------------
    # 9. Create output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 10. Save raw SHAP values
    # ---------------------------------------------------------

    np.save(
        OUTPUT_DIR / "shap_values.npy",
        shap_values,
    )

    # ---------------------------------------------------------
    # 11. Save processed feature matrix
    # ---------------------------------------------------------

    np.save(
        OUTPUT_DIR / "processed_features.npy",
        X_processed,
    )

    # ---------------------------------------------------------
    # 12. Save feature names
    # ---------------------------------------------------------

    pd.Series(
        feature_names
    ).to_csv(
        OUTPUT_DIR / "feature_names.csv",
        index=False,
        header=["feature"],
    )

    # ---------------------------------------------------------
    # 13. Calculate global SHAP importance
    # ---------------------------------------------------------

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # 14. Save global SHAP importance
    # ---------------------------------------------------------

    importance.to_csv(
        OUTPUT_DIR / "global_shap_importance.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 15. Display top 20 features
    # ---------------------------------------------------------

    print("\nTop 20 SHAP features:")

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 16. Completion
    # ---------------------------------------------------------

    print(
        "\nSHAP analysis completed successfully."
    )

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()