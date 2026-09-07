import pandas as pd


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records from the dataset.
    """

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    after = len(df)

    print(f"Removed {before - after} duplicate rows.")

    return df