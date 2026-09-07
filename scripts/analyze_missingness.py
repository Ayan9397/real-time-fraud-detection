from pathlib import Path

import pandas as pd


DATA_FILE = Path(
    "data/processed/train.csv"
)


def main():

    print("=" * 70)
    print("MISSINGNESS STRUCTURE ANALYSIS")
    print("=" * 70)

    print("\nLoading dataset...")

    # Only load columns needed for this analysis.
    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print("\nCalculating missingness...")

    missing_rate = (
        df.isna()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("TOP 30 MOST-MISSING FEATURES")
    print("=" * 70)

    for feature, rate in missing_rate.head(30).items():

        print(
            f"{feature:<15} "
            f"{rate:.2%}"
        )

    # ---------------------------------------------------------
    # Missing feature count per transaction
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING FEATURES PER TRANSACTION")
    print("=" * 70)

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "TransactionID",
            "isFraud",
        ]
    ]

    missing_count = (
        df[feature_columns]
        .isna()
        .sum(axis=1)
    )

    print(
        missing_count.describe()
    )

    # ---------------------------------------------------------
    # Relationship between missingness and fraud
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FRAUD RATE BY MISSING-FEATURE COUNT")
    print("=" * 70)

    analysis = pd.DataFrame(
        {
            "missing_count":
                missing_count,
            "isFraud":
                df["isFraud"],
        }
    )

    grouped = (
        analysis
        .groupby("missing_count")
        .agg(
            transactions=(
                "isFraud",
                "count",
            ),
            fraud_count=(
                "isFraud",
                "sum",
            ),
            fraud_rate=(
                "isFraud",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["transactions"] >= 100
    ]

    print(
        grouped
        .to_string(
            index=False,
            formatters={
                "fraud_rate":
                    "{:.4%}".format
            },
        )
    )

    # ---------------------------------------------------------
    # V feature missingness
    # ---------------------------------------------------------

    v_features = [
        column
        for column in df.columns
        if column.startswith("V")
        and column[1:].isdigit()
    ]

    v_missing = (
        df[v_features]
        .isna()
        .sum(axis=1)
    )

    print("\n" + "=" * 70)
    print("V-FEATURE MISSINGNESS")
    print("=" * 70)

    print(
        v_missing.describe()
    )

    v_analysis = pd.DataFrame(
        {
            "v_missing":
                v_missing,
            "isFraud":
                df["isFraud"],
        }
    )

    v_grouped = (
        v_analysis
        .groupby("v_missing")
        .agg(
            transactions=(
                "isFraud",
                "count",
            ),
            fraud_count=(
                "isFraud",
                "sum",
            ),
            fraud_rate=(
                "isFraud",
                "mean",
            ),
        )
        .reset_index()
    )

    v_grouped = v_grouped[
        v_grouped["transactions"] >= 100
    ]

    print(
        v_grouped
        .to_string(
            index=False,
            formatters={
                "fraud_rate":
                    "{:.4%}".format
            },
        )
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()