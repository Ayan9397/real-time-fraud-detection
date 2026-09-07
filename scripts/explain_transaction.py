from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("models/xgboost_missing_model.joblib")
PREPROCESSOR_PATH = Path("models/xgboost_missing_preprocessor.joblib")
VALID_PATH = Path("data/processed/valid.csv")
OUTPUT_DIR = Path("models/shap")

RANDOM_STATE = 42


def add_missing_indicators(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create the same missingness features used during training."""

    df = df.copy()

    indicators = pd.DataFrame(
        {
            f"{column}_missing": df[column].isna().astype("int8")
            for column in feature_columns
        },
        index=df.index,
    )

    return pd.concat(
        [df, indicators],
        axis=1,
    )


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

    # ---------------------------------------------------------
    # 1. Create model features
    # ---------------------------------------------------------

    X = df[feature_columns].copy()

    X = add_missing_indicators(
        X,
        feature_columns,
    )

    # Match exact training schema.
    expected_columns = list(
        preprocessor.feature_names_in_
    )

    X = X[expected_columns]

    # ---------------------------------------------------------
    # 2. Select one deterministic validation transaction
    # ---------------------------------------------------------

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    selected_index = rng.randint(
        0,
        len(X),
    )

    transaction = X.iloc[
        [selected_index]
    ].copy()

    original_transaction = df.iloc[
        selected_index
    ]

    transaction_id = original_transaction[
        id_column
    ]

    actual_label = original_transaction[
        target
    ]

    print("\nSelected transaction:")
    print(
        f"TransactionID: {transaction_id}"
    )
    print(
        f"Actual label: {actual_label}"
    )

    # ---------------------------------------------------------
    # 3. Preprocess transaction
    # ---------------------------------------------------------

    print("\nApplying preprocessing...")

    processed = preprocessor.transform(
        transaction
    )

    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    processed = np.asarray(
        processed
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    # ---------------------------------------------------------
    # 4. Predict fraud probability
    # ---------------------------------------------------------

    fraud_probability = float(
        model.predict_proba(
            processed
        )[0, 1]
    )

    print(
        f"\nFraud probability: {fraud_probability:.6f}"
    )

    # Project operating threshold selected earlier.
    threshold = 0.60

    decision = (
        "FRAUD REVIEW"
        if fraud_probability >= threshold
        else "LEGITIMATE"
    )

    print(
        f"Decision at threshold {threshold:.2f}: {decision}"
    )

    # ---------------------------------------------------------
    # 5. Calculate SHAP values
    # ---------------------------------------------------------

    print("\nCalculating SHAP explanation...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        processed
    )

    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    # First/only transaction.
    values = shap_values[0]

    # ---------------------------------------------------------
    # 6. Build explanation table
    # ---------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": values,
            "absolute_shap": np.abs(values),
        }
    )

    explanation["direction"] = np.where(
        explanation["shap_value"] > 0,
        "toward_fraud",
        "toward_legitimate",
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # 7. Save explanation
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "individual_transaction_explanation.csv"
    )

    explanation.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 8. Display strongest contributors
    # ---------------------------------------------------------

    print(
        "\nTop 10 factors influencing this prediction:"
    )

    print(
        explanation.head(10).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 9. Separate fraud / legitimate contributors
    # ---------------------------------------------------------

    fraud_factors = explanation[
        explanation["shap_value"] > 0
    ].head(10)

    legitimate_factors = explanation[
        explanation["shap_value"] < 0
    ].head(10)

    print(
        "\nTop factors pushing toward FRAUD:"
    )

    if fraud_factors.empty:
        print("None")

    else:
        print(
            fraud_factors[
                [
                    "feature",
                    "shap_value",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nTop factors pushing toward LEGITIMATE:"
    )

    if legitimate_factors.empty:
        print("None")

    else:
        print(
            legitimate_factors[
                [
                    "feature",
                    "shap_value",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nIndividual transaction explanation completed."
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()