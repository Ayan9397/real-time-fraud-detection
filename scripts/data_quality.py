from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")

TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"


def check_transaction_ids() -> None:
    """Check TransactionID integrity."""

    print("=" * 70)
    print("TRANSACTION ID INTEGRITY")
    print("=" * 70)

    duplicate_count = 0
    total_rows = 0
    min_id = None
    max_id = None

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        usecols=["TransactionID"],
        chunksize=50_000,
    ):
        total_rows += len(chunk)

        duplicate_count += chunk["TransactionID"].duplicated().sum()

        chunk_min = chunk["TransactionID"].min()
        chunk_max = chunk["TransactionID"].max()

        if min_id is None or chunk_min < min_id:
            min_id = chunk_min

        if max_id is None or chunk_max > max_id:
            max_id = chunk_max

    print(f"Total rows: {total_rows:,}")
    print(f"Minimum TransactionID: {min_id:,}")
    print(f"Maximum TransactionID: {max_id:,}")
    print(f"Duplicate IDs within chunks: {duplicate_count:,}")


def check_target() -> None:
    """Check target values and counts."""

    print("\n" + "=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)

    target_counts = {}

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        usecols=["isFraud"],
        chunksize=50_000,
    ):
        counts = chunk["isFraud"].value_counts()

        for value, count in counts.items():
            target_counts[value] = (
                target_counts.get(value, 0) + count
            )

    print("Target values:")

    for value, count in sorted(target_counts.items()):
        print(f"  {value}: {count:,}")


def check_transaction_time() -> None:
    """Inspect TransactionDT."""

    print("\n" + "=" * 70)
    print("TRANSACTION TIME")
    print("=" * 70)

    result = {
        "min": None,
        "max": None,
    }

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        usecols=["TransactionDT"],
        chunksize=50_000,
    ):
        chunk_min = chunk["TransactionDT"].min()
        chunk_max = chunk["TransactionDT"].max()

        if result["min"] is None or chunk_min < result["min"]:
            result["min"] = chunk_min

        if result["max"] is None or chunk_max > result["max"]:
            result["max"] = chunk_max

    print(f"Minimum TransactionDT: {result['min']:,}")
    print(f"Maximum TransactionDT: {result['max']:,}")


def check_transaction_amount() -> None:
    """Inspect transaction amounts."""

    print("\n" + "=" * 70)
    print("TRANSACTION AMOUNT")
    print("=" * 70)

    minimum = float("inf")
    maximum = float("-inf")
    total = 0
    count = 0
    zero_or_negative = 0

    for chunk in pd.read_csv(
        TRANSACTION_FILE,
        usecols=["TransactionAmt"],
        chunksize=50_000,
    ):
        series = chunk["TransactionAmt"].dropna()

        if series.empty:
            continue

        minimum = min(minimum, series.min())
        maximum = max(maximum, series.max())

        total += series.sum()
        count += len(series)

        zero_or_negative += (series <= 0).sum()

    mean = total / count

    print(f"Minimum amount: {minimum:.4f}")
    print(f"Maximum amount: {maximum:.4f}")
    print(f"Mean amount: {mean:.4f}")
    print(f"Zero/negative amounts: {zero_or_negative:,}")


def check_identity_ids() -> None:
    """Check identity TransactionID uniqueness."""

    print("\n" + "=" * 70)
    print("IDENTITY ID INTEGRITY")
    print("=" * 70)

    identity = pd.read_csv(
        IDENTITY_FILE,
        usecols=["TransactionID"],
    )

    duplicate_count = identity["TransactionID"].duplicated().sum()

    print(f"Identity rows: {len(identity):,}")
    print(f"Duplicate identity IDs: {duplicate_count:,}")


def main() -> None:
    check_transaction_ids()
    check_target()
    check_transaction_time()
    check_transaction_amount()
    check_identity_ids()


if __name__ == "__main__":
    main()