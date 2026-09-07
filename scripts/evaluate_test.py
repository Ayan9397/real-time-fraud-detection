from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TEST_FILE = Path("data/processed/test.csv")

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

    total = len(y_true)

    false_positive_rate = fp / (fp + tn)

    fraud_detection_rate = recall

    print("\n" + "-" * 70)
    print(f"THRESHOLD: {threshold:.2f}")
    print("-" * 70)

    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1:                  {f1:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4%}")
    print(f"Fraud Detection:     {fraud_detection_rate:.4%}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)


def main():

    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "\nIMPORTANT: This is the final unseen test evaluation."
    )

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    print("\nLoading test dataset...")

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        "Test shape:",
        test_df.shape,
    )

    # ---------------------------------------------------------
    # Separate target
    # ---------------------------------------------------------

    y_test = test_df[TARGET]

    X_test = test_df.drop(
        columns=[
            TARGET,
            ID_COLUMN,
        ]
    ).copy()

    print("\nTest target distribution:")
    print(y_test.value_counts())

    print(
        "\nTest fraud rate:",
        f"{y_test.mean():.4%}",
    )

    # ---------------------------------------------------------
    # Create missingness indicators
    # ---------------------------------------------------------

    print(
        "\nCreating missingness indicators..."
    )

    missing_test = (
        X_test.isna()
        .astype("int8")
    )

    missing_test.columns = [
        f"{column}_missing"
        for column in missing_test.columns
    ]

    X_test = pd.concat(
        [
            X_test,
            missing_test,
        ],
        axis=1,
    )

    print(
        "Features after indicators:",
        X_test.shape[1],
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print(
        "\nLoading champion model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    # ---------------------------------------------------------
    # Transform
    # ---------------------------------------------------------

    print(
        "Transforming test data..."
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "Processed test shape:",
        X_test_processed.shape,
    )

    # ---------------------------------------------------------
    # Generate probabilities
    # ---------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    # ---------------------------------------------------------
    # Threshold-independent metrics
    # ---------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("THRESHOLD-INDEPENDENT TEST METRICS")
    print("=" * 70)

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    # ---------------------------------------------------------
    # Evaluate selected thresholds
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST PERFORMANCE AT DIFFERENT THRESHOLDS")
    print("=" * 70)

    for threshold in [
        0.50,
        0.60,
        0.70,
    ]:

        evaluate_threshold(
            y_test,
            probabilities,
            threshold,
        )

    # ---------------------------------------------------------
    # Save test predictions
    # ---------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "TransactionID": test_df[
                ID_COLUMN
            ],
            "actual_isFraud": y_test,
            "fraud_probability": probabilities,
        }
    )

    predictions_file = Path(
        "models/test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("TEST PREDICTIONS SAVED")
    print("=" * 70)

    print(
        f"File: {predictions_file}"
    )

    print("\nFinal test evaluation complete.")


if __name__ == "__main__":
    main()