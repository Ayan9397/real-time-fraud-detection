from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


VALID_FILE = Path("data/processed/valid.csv")

MODEL_FILE = Path(
    "models/xgboost_missing_model.joblib"
)

PREPROCESSOR_FILE = Path(
    "models/xgboost_missing_preprocessor.joblib"
)

TARGET = "isFraud"
ID_COLUMN = "TransactionID"


def evaluate_threshold(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def main():

    print("=" * 70)
    print("FRAUD DETECTION THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    print("\nLoading validation dataset...")

    valid_df = pd.read_csv(
        VALID_FILE
    )

    print(
        "Validation shape:",
        valid_df.shape,
    )

    y_valid = valid_df[TARGET]

    X_valid = valid_df.drop(
        columns=[
            TARGET,
            ID_COLUMN,
        ]
    ).copy()

    # ---------------------------------------------------------
    # Recreate missingness indicators
    # ---------------------------------------------------------

    print(
        "\nCreating missingness indicators..."
    )

    missing_valid = (
        X_valid.isna()
        .astype("int8")
    )

    missing_valid.columns = [
        f"{column}_missing"
        for column in missing_valid.columns
    ]

    X_valid = pd.concat(
        [
            X_valid,
            missing_valid,
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # Load preprocessing and model
    # ---------------------------------------------------------

    print(
        "\nLoading model and preprocessor..."
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    # ---------------------------------------------------------
    # Transform validation data
    # ---------------------------------------------------------

    print(
        "Transforming validation data..."
    )

    X_valid_processed = (
        preprocessor.transform(
            X_valid
        )
    )

    print(
        "Processed shape:",
        X_valid_processed.shape,
    )

    # ---------------------------------------------------------
    # Generate probabilities
    # ---------------------------------------------------------

    print(
        "\nGenerating fraud probabilities..."
    )

    probabilities = model.predict_proba(
        X_valid_processed
    )[:, 1]

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    print(
        f"Validation PR-AUC: {pr_auc:.4f}"
    )

    # ---------------------------------------------------------
    # Test thresholds
    # ---------------------------------------------------------

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            y_valid,
            probabilities,
            threshold,
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "precision":
                    "{:.4f}".format,
                "recall":
                    "{:.4f}".format,
                "f1":
                    "{:.4f}".format,
            },
        )
    )

    # ---------------------------------------------------------
    # Best F1
    # ---------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_f1['f1']:.4f}"
    )

    # ---------------------------------------------------------
    # Best precision while maintaining recall >= 50%
    # ---------------------------------------------------------

    recall_50 = results_df[
        results_df["recall"] >= 0.50
    ]

    if not recall_50.empty:

        best_precision_50 = (
            recall_50.loc[
                recall_50[
                    "precision"
                ].idxmax()
            ]
        )

        print("\n" + "=" * 70)
        print(
            "BEST PRECISION "
            "WITH RECALL >= 50%"
        )
        print("=" * 70)

        print(
            f"Threshold: "
            f"{best_precision_50['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_precision_50['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_precision_50['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{best_precision_50['f1']:.4f}"
        )

    # ---------------------------------------------------------
    # Best recall while maintaining precision >= 50%
    # ---------------------------------------------------------

    precision_50 = results_df[
        results_df["precision"] >= 0.50
    ]

    if not precision_50.empty:

        best_recall_50 = (
            precision_50.loc[
                precision_50[
                    "recall"
                ].idxmax()
            ]
        )

        print("\n" + "=" * 70)
        print(
            "BEST RECALL "
            "WITH PRECISION >= 50%"
        )
        print("=" * 70)

        print(
            f"Threshold: "
            f"{best_recall_50['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_recall_50['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_recall_50['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{best_recall_50['f1']:.4f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_file = Path(
        "models/threshold_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"File: {output_file}"
    )


if __name__ == "__main__":
    main()