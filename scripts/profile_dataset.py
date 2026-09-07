from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")

TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"


def profile_transaction_data() -> None:
    """Profile the transaction dataset using chunks."""

    print("=" * 70)
    print("TRANSACTION DATA PROFILE")
    print("=" * 70)

    # Read only the header first.
    columns = pd.read_csv(
        TRANSACTION_FILE,
        nrows=0
    ).columns.tolist()

    print(f"\nTotal columns: {len(columns)}")

    numeric_columns = []
    categorical_columns = []

    # Read a sample to determine initial data types.
    sample = pd.read_csv(
        TRANSACTION_FILE,
        nrows=10_000
    )

    for column in sample.columns:
        if pd.api.types.is_numeric_dtype(sample[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")

    print("\nCategorical columns:")
    for column in categorical_columns:
        unique_count = sample[column].nunique(dropna=True)

        print(
            f"  {column:<20} "
            f"unique values: {unique_count}"
        )

    print("\nNumeric columns:")
    print(
        "  "
        + ", ".join(numeric_columns[:30])
    )

    if len(numeric_columns) > 30:
        print(
            f"  ... and {len(numeric_columns) - 30} more"
        )


def profile_missing_values() -> None:
    """Calculate missing-value percentages using chunks."""

    print("\n" + "=" * 70)
    print("MISSING VALUE PROFILE")
    print("=" * 70)

    missing_counts = {}
    total_rows = 0

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        chunksize=50_000
    ):
        total_rows += len(chunk)

        missing = chunk.isna().sum()

        for column, count in missing.items():
            missing_counts[column] = (
                missing_counts.get(column, 0) + count
            )

    missing_df = pd.DataFrame(
        {
            "column": list(missing_counts.keys()),
            "missing_count": list(missing_counts.values()),
        }
    )

    missing_df["missing_percentage"] = (
        missing_df["missing_count"]
        / total_rows
        * 100
    )

    missing_df = missing_df.sort_values(
        "missing_percentage",
        ascending=False
    )

    print("\nTop 30 columns by missing percentage:\n")

    print(
        missing_df.head(30).to_string(index=False)
    )

    return missing_df


def profile_identity_coverage() -> None:
    """Calculate how many transactions have identity information."""

    print("\n" + "=" * 70)
    print("IDENTITY DATA COVERAGE")
    print("=" * 70)

    transaction_ids = set()

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        usecols=["TransactionID"],
        chunksize=50_000
    ):
        transaction_ids.update(
            chunk["TransactionID"]
        )

    identity_ids = pd.read_csv(
        IDENTITY_FILE,
        usecols=["TransactionID"]
    )

    identity_id_set = set(
        identity_ids["TransactionID"]
    )

    matched = len(
        transaction_ids.intersection(identity_id_set)
    )

    total = len(transaction_ids)

    percentage = matched / total * 100

    print(f"Total transactions: {total:,}")
    print(f"Identity records: {len(identity_ids):,}")
    print(f"Matched transactions: {matched:,}")
    print(f"Identity coverage: {percentage:.2f}%")


def main() -> None:
    profile_transaction_data()
    profile_missing_values()
    profile_identity_coverage()


if __name__ == "__main__":
    main()