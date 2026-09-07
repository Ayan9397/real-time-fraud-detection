from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")


def load_csv(file_name: str) -> pd.DataFrame:
    """
    Load a CSV file from the raw data directory.
    """

    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print(f"Loading dataset: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

    return df


def load_fraud_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transaction and identity datasets.
    """

    transactions = load_csv("train_transaction.csv")
    identity = load_csv("train_identity.csv")

    return transactions, identity


if __name__ == "__main__":
    print("Data ingestion module ready.")