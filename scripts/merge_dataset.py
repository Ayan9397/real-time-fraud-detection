from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"

OUTPUT_FILE = PROCESSED_DATA_DIR / "fraud_dataset.csv"


def merge_datasets() -> None:
    """Merge transaction and identity datasets using TransactionID."""

    print("=" * 70)
    print("LOADING TRANSACTION DATA")
    print("=" * 70)

    transactions = pd.read_csv(TRANSACTION_FILE)

    print(f"Transaction shape: {transactions.shape}")

    print("\n" + "=" * 70)
    print("LOADING IDENTITY DATA")
    print("=" * 70)

    identity = pd.read_csv(IDENTITY_FILE)

    print(f"Identity shape: {identity.shape}")

    print("\n" + "=" * 70)
    print("CHECKING JOIN KEY")
    print("=" * 70)

    if transactions["TransactionID"].duplicated().any():
        raise ValueError(
            "Duplicate TransactionID found in transaction data."
        )

    if identity["TransactionID"].duplicated().any():
        raise ValueError(
            "Duplicate TransactionID found in identity data."
        )

    print("TransactionID integrity check passed.")

    print("\n" + "=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_identity"),
    )

    print(f"Merged shape: {merged.shape}")

    print("\n" + "=" * 70)
    print("IDENTITY COVERAGE AFTER MERGE")
    print("=" * 70)

    identity_matches = merged["id_01"].notna().sum()
    total_transactions = len(merged)

    coverage = identity_matches / total_transactions * 100

    print(f"Transactions with identity data: {identity_matches:,}")
    print(f"Transactions without identity data: {total_transactions - identity_matches:,}")
    print(f"Identity coverage: {coverage:.2f}%")

    print("\n" + "=" * 70)
    print("TARGET PRESERVATION")
    print("=" * 70)

    print("Target distribution after merge:")

    print(
        merged["isFraud"]
        .value_counts()
        .sort_index()
    )

    if len(merged) != len(transactions):
        raise ValueError(
            "Transaction row count changed after merge."
        )

    print("\nTransaction row count preserved.")

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("SAVING MERGED DATASET")
    print("=" * 70)

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"Final dataset shape: {merged.shape}")


if __name__ == "__main__":
    merge_datasets()