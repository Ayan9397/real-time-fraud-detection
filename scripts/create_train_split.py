
from pathlib import Path

import pandas as pd

DATA_FILE = Path("data/processed/fraud_dataset.csv")

OUTPUT_DIR = Path("data/processed")


def main():

    print("Loading merged dataset...")

    df = pd.read_csv(DATA_FILE)

    print(df.shape)

    df = df.sort_values("TransactionDT")

    train_end = int(len(df) * 0.70)
    valid_end = int(len(df) * 0.85)

    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    valid.to_csv(
        OUTPUT_DIR / "valid.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print("\nTemporal split completed.")

    print(f"Train: {train.shape}")
    print(f"Valid: {valid.shape}")
    print(f"Test : {test.shape}")


if __name__ == "__main__":
    main()