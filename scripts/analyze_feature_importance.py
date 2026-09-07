from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/xgboost_fraud_model.joblib"
)

DATA_PATH = Path(
    "data/processed/train.csv"
)


def main():

    print("=" * 70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 70)

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded.")

    print("\nLoading feature names...")

    df = pd.read_csv(
        DATA_PATH,
        nrows=1,
    )

    feature_names = [
        column
        for column in df.columns
        if column not in [
            "TransactionID",
            "isFraud",
        ]
    ]

    print(
        f"Feature count: "
        f"{len(feature_names)}"
    )

    importance = model.feature_importances_

    if len(importance) != len(feature_names):
        raise ValueError(
            "Feature count does not match "
            "model importance count."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("TOP 30 FEATURES")
    print("=" * 70)

    print(
        importance_df.head(30).to_string(
            index=False
        )
    )

    output_path = Path(
        "models/feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        f"Feature importance saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()