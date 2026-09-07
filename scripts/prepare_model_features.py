from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")

TARGET = "isFraud"


def select_features(df: pd.DataFrame) -> list[str]:
    """Select features for the fraud model."""

    features = []

    # Transaction features
    transaction_features = [
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
    ]

    features.extend(
        column
        for column in transaction_features
        if column in df.columns
    )

    # Card features
    features.extend(
        column
        for column in df.columns
        if column.startswith("card")
        and column[4:].isdigit()
    )

    # Address and distance
    features.extend(
        column
        for column in [
            "addr1",
            "addr2",
            "dist1",
            "dist2",
        ]
        if column in df.columns
    )

    # Email
    features.extend(
        column
        for column in [
            "P_emaildomain",
            "R_emaildomain",
        ]
        if column in df.columns
    )

    # C features
    features.extend(
        column
        for column in df.columns
        if column.startswith("C")
        and column[1:].isdigit()
    )

    # D features
    features.extend(
        column
        for column in df.columns
        if column.startswith("D")
        and column[1:].isdigit()
    )

    # M features
    features.extend(
        column
        for column in df.columns
        if column.startswith("M")
        and column[1:].isdigit()
    )

    # V features
    features.extend(
        column
        for column in df.columns
        if column.startswith("V")
        and column[1:].isdigit()
    )

    # Identity features
    features.extend(
        column
        for column in df.columns
        if column.startswith("id_")
        and column[3:].isdigit()
    )

    # Device information
    for column in [
        "DeviceType",
        "DeviceInfo",
    ]:
        if column in df.columns:
            features.append(column)

    # Remove duplicates while preserving order.
    features = list(dict.fromkeys(features))

    return features


def main():

    print("=" * 70)
    print("MODEL FEATURE PREPARATION")
    print("=" * 70)

    print("Loading training dataset...")

    df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    print(
        f"Original shape: {df.shape}"
    )

    features = select_features(df)

    print(
        f"Selected features: {len(features)}"
    )

    print("\nExcluded columns:")

    excluded = [
        column
        for column in df.columns
        if column not in features
        and column != TARGET
    ]

    for column in excluded:
        print(f"  {column}")

    # Build feature groups independently.
    transaction_group = [
        c for c in features
        if c in [
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
        ]
    ]

    card_group = [
        c for c in features
        if c.startswith("card")
    ]

    c_group = [
        c for c in features
        if c.startswith("C")
        and c[1:].isdigit()
    ]

    d_group = [
        c for c in features
        if c.startswith("D")
        and c[1:].isdigit()
    ]

    m_group = [
        c for c in features
        if c.startswith("M")
        and c[1:].isdigit()
    ]

    v_group = [
        c for c in features
        if c.startswith("V")
        and c[1:].isdigit()
    ]

    identity_group = [
        c for c in features
        if c.startswith("id_")
        and c[3:].isdigit()
    ]

    known_features = set(
        transaction_group
        + card_group
        + c_group
        + d_group
        + m_group
        + v_group
        + identity_group
        + [
            "addr1",
            "addr2",
            "dist1",
            "dist2",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "DeviceInfo",
        ]
    )

    other_group = [
        c for c in features
        if c not in known_features
    ]

    print("\n" + "=" * 70)
    print("SELECTED FEATURE GROUPS")
    print("=" * 70)

    print(
        f"Transaction: {len(transaction_group)}"
    )

    print(
        f"Card:        {len(card_group)}"
    )

    print(
        f"C features:  {len(c_group)}"
    )

    print(
        f"D features:  {len(d_group)}"
    )

    print(
        f"M features:  {len(m_group)}"
    )

    print(
        f"V features:  {len(v_group)}"
    )

    print(
        f"Identity:    {len(identity_group)}"
    )

    print(
        f"Other:       {len(other_group)}"
    )

    print("\n" + "=" * 70)
    print("FEATURE AUDIT")
    print("=" * 70)

    print(
        f"Total selected features: {len(features)}"
    )

    print(
        f"Target excluded: {TARGET}"
    )

    print(
        "TransactionID excluded: "
        f"{'TransactionID' not in features}"
    )

    print("\nFeature preparation complete.")


if __name__ == "__main__":
    main()