import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Perform basic validation checks on the input DataFrame.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    missing_percentage = df.isnull().mean() * 100

    print("Dataset shape:", df.shape)

    print("\nMissing values (%):")

    missing_values = missing_percentage[missing_percentage > 0]

    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values)

    print("\nValidation completed successfully.")