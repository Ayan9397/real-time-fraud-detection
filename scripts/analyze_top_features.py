from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/processed/train.csv")


TOP_FEATURES = [
    "V244",
    "V204",
    "V243",
    "V56",
    "V280",
    "V187",
    "V77",
    "C3",
    "V173",
    "V232",
]


def analyze_feature(df: pd.DataFrame, feature: str) -> None:

    print("\n" + "=" * 70)
    print(f"FEATURE: {feature}")
    print("=" * 70)

    series = df[feature]

    missing = series.isna().sum()
    missing_rate = missing / len(df) * 100

    print(
        f"Missing: {missing:,} "
        f"({missing_rate:.2f}%)"
    )

    print(
        f"Unique values: "
        f"{series.nunique(dropna=True):,}"
    )

    # Compare fraud rate for rows where
    # the feature exists vs where it is missing.
    missing_mask = series.isna()

    fraud_missing = (
        df.loc[missing_mask, "isFraud"].mean()
        if missing_mask.any()
        else 0
    )

    fraud_present = (
        df.loc[~missing_mask, "isFraud"].mean()
        if (~missing_mask).any()
        else 0
    )

    print(
        f"Fraud rate when missing: "
        f"{fraud_missing:.4%}"
    )

    print(
        f"Fraud rate when present: "
        f"{fraud_present:.4%}"
    )

    if pd.api.types.is_numeric_dtype(series):

        print("\nNumeric summary:")

        print(
            series.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ]
            )
        )


def main():

    print("=" * 70)
    print("TOP FEATURE ANALYSIS")
    print("=" * 70)

    print("\nLoading training data...")

    columns = TOP_FEATURES + ["isFraud"]

    df = pd.read_csv(
        DATA_FILE,
        usecols=columns,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    for feature in TOP_FEATURES:
        analyze_feature(
            df,
            feature,
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()