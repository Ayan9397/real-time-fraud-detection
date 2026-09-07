from pathlib import Path

import mlflow
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

VALIDATION_DATA = PROJECT_ROOT / "data" / "processed" / "valid.csv"

REGISTERED_MODEL = "fraud_detection_xgboost"
MODEL_VERSION = "2"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VERIFYING REGISTERED MLflow MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

    print("\nConnecting to MLflow...")
    print("Tracking URI:", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)

    # --------------------------------------------------------
    # Model URI
    # --------------------------------------------------------

    model_uri = (
        f"models:/{REGISTERED_MODEL}/{MODEL_VERSION}"
    )

    print("\nLoading registered model:")
    print(model_uri)

    # --------------------------------------------------------
    # Load deployable model
    # --------------------------------------------------------

    model = mlflow.pyfunc.load_model(model_uri)

    print("\nModel loaded successfully.")

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("\nLoading validation data...")

    if not VALIDATION_DATA.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_DATA}"
        )

    df = pd.read_csv(VALIDATION_DATA)

    print("Validation shape:", df.shape)

    # --------------------------------------------------------
    # Select a small sample
    # --------------------------------------------------------

    sample = df.head(5).copy()

    print("\nSample transactions:")
    print(sample[["TransactionID", "isFraud", "TransactionAmt"]])

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    X_sample = sample.drop(
        columns=["isFraud"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nRunning inference...")

    predictions = model.predict(X_sample)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nPrediction results:")
    print("-" * 70)

    results = pd.DataFrame(predictions)

    if "TransactionID" in sample.columns:
        results.insert(
            0,
            "TransactionID",
            sample["TransactionID"].values
        )

    if "isFraud" in sample.columns:
        results.insert(
            1,
            "actual_isFraud",
            sample["isFraud"].values
        )

    print(results.to_string(index=False))

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_columns = {
        "fraud_probability",
        "fraud_prediction",
        "decision",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    if not results["fraud_probability"].between(
        0,
        1
    ).all():

        raise ValueError(
            "Fraud probabilities must be between 0 and 1."
        )

    print("\n" + "=" * 70)
    print("REGISTERED MODEL VERIFICATION PASSED")
    print("=" * 70)

    print("\nModel:")
    print(REGISTERED_MODEL)

    print("Version:")
    print(MODEL_VERSION)

    print("Inference:")
    print("SUCCESS")

    print("\nThe registered MLflow model can accept")
    print("raw transaction data and return fraud predictions.")

    print("=" * 70)


if __name__ == "__main__":
    main()