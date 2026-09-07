from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")

TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"


def inspect_file(file_path: Path, sample_rows: int = 5) -> None:
    """Inspect a CSV without loading the complete file into memory."""

    print("=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    print(f"File size: {file_size_mb:.2f} MB")

    # Read only a small sample.
    sample = pd.read_csv(file_path, nrows=sample_rows)

    print(f"\nNumber of columns: {len(sample.columns)}")

    print("\nColumns:")
    for column in sample.columns:
        print(f"  - {column}")

    print("\nSample:")
    print(sample.head())

    print("\nData types:")
    print(sample.dtypes)


def inspect_target() -> None:
    """Inspect the fraud target using chunks."""

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    fraud_counts = {}

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        usecols=["isFraud"],
        chunksize=50_000,
    ):
        counts = chunk["isFraud"].value_counts()

        for value, count in counts.items():
            fraud_counts[value] = fraud_counts.get(value, 0) + count

    total = sum(fraud_counts.values())

    for value, count in sorted(fraud_counts.items()):
        percentage = (count / total) * 100

        label = "Fraud" if value == 1 else "Legitimate"

        print(
            f"{label}: {count:,} "
            f"({percentage:.4f}%)"
        )

    print(f"\nTotal transactions: {total:,}")


def main() -> None:
    inspect_file(TRANSACTION_FILE)
    inspect_file(IDENTITY_FILE)
    inspect_target()


if __name__ == "__main__":
    main()