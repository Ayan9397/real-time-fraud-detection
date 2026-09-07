from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/processed/fraud_dataset.csv")


def audit_features() -> None:
    print("=" * 70)
    print("LOADING PROCESSED DATASET")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset shape: {df.shape}")

    print("\n" + "=" * 70)
    print("TARGET")
    print("=" * 70)

    print("Target column: isFraud")

    print(
        df["isFraud"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("IDENTIFIER COLUMNS")
    print("=" * 70)

    identifier_candidates = [
        column
        for column in df.columns
        if "id" in column.lower()
    ]

    for column in identifier_candidates:
        print(
            f"{column}: "
            f"{df[column].nunique(dropna=True):,} unique values"
        )

    print("\n" + "=" * 70)
    print("NUMERIC / CATEGORICAL FEATURES")
    print("=" * 70)

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = df.select_dtypes(
        exclude=["number"]
    ).columns

    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")

    print("\nCategorical columns:")

    for column in categorical_columns:
        print(
            f"  {column}: "
            f"{df[column].nunique(dropna=True):,} unique values"
        )

    print("\n" + "=" * 70)
    print("HIGH-CARDINALITY FEATURES")
    print("=" * 70)

    for column in df.columns:

        unique_count = df[column].nunique(
            dropna=True
        )

        if unique_count > 1000:

            print(
                f"{column}: "
                f"{unique_count:,} unique values"
            )

    print("\n" + "=" * 70)
    print("TARGET LEAKAGE CHECK")
    print("=" * 70)

    leakage_candidates = []

    for column in df.columns:

        if column == "isFraud":
            continue

        if column.lower() in [
            "target",
            "label",
            "fraud",
            "is_fraud",
        ]:
            leakage_candidates.append(column)

    if leakage_candidates:
        print("Potential target leakage columns:")

        for column in leakage_candidates:
            print(f"  {column}")

    else:
        print("No obvious target leakage column names detected.")

    print("\n" + "=" * 70)
    print("FEATURE AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    audit_features()