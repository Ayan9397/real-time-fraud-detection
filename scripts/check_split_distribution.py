from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")


def inspect_split(name: str) -> None:
    file_path = DATA_DIR / f"{name}.csv"

    df = pd.read_csv(
        file_path,
        usecols=["TransactionDT", "isFraud"],
    )

    total = len(df)
    fraud = df["isFraud"].sum()
    fraud_rate = fraud / total * 100

    print("=" * 70)
    print(f"{name.upper()} SET")
    print("=" * 70)

    print(f"Rows: {total:,}")
    print(f"Fraud: {fraud:,}")
    print(f"Legitimate: {total - fraud:,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")

    print(
        f"TransactionDT range: "
        f"{df['TransactionDT'].min():,} "
        f"-> "
        f"{df['TransactionDT'].max():,}"
    )


def main() -> None:
    inspect_split("train")
    inspect_split("valid")
    inspect_split("test")


if __name__ == "__main__":
    main()