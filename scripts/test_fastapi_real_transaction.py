from pathlib import Path
import json
import urllib.request
import urllib.error

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "valid.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

TRANSACTION_ID = 3400379

EXPECTED_MLFLOW_PROBABILITY = 0.985441

TOLERANCE = 0.00001


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FASTAPI REAL TRANSACTION VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    if not VALIDATION_DATA.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_DATA}"
        )

    df = pd.read_csv(VALIDATION_DATA)

    print("Validation shape:", df.shape)

    # --------------------------------------------------------
    # Find target transaction
    # --------------------------------------------------------

    transaction_rows = df[
        df["TransactionID"] == TRANSACTION_ID
    ]

    if transaction_rows.empty:
        raise ValueError(
            f"TransactionID {TRANSACTION_ID} "
            "was not found in valid.csv."
        )

    if len(transaction_rows) > 1:
        raise ValueError(
            f"TransactionID {TRANSACTION_ID} "
            "appears more than once."
        )

    transaction = transaction_rows.iloc[0]

    print("\nTransaction found.")

    print(
        "TransactionID:",
        transaction["TransactionID"]
    )

    print(
        "Actual isFraud:",
        transaction["isFraud"]
    )

    print(
        "TransactionAmt:",
        transaction["TransactionAmt"]
    )

    # --------------------------------------------------------
    # Convert row to JSON-compatible dictionary
    # --------------------------------------------------------

    payload_df = pd.DataFrame(
        [transaction.drop(labels=["isFraud"])]
    )

    # Convert NaN values to None so they become JSON null.

    payload_df = payload_df.astype(object)

    payload_df = payload_df.where(
        pd.notna(payload_df),
        None
    )

    payload = payload_df.iloc[0].to_dict()

    # --------------------------------------------------------
    # Convert NumPy values into native Python values
    # --------------------------------------------------------

    for key, value in payload.items():

        if hasattr(value, "item"):
            payload[key] = value.item()

    # --------------------------------------------------------
    # Display payload information
    # --------------------------------------------------------

    print(
        "\nFeatures being sent to FastAPI:",
        len(payload)
    )

    print(
        "Expected original model features:",
        432
    )

    # --------------------------------------------------------
    # Send HTTP request
    # --------------------------------------------------------

    print("\nSending complete transaction to FastAPI...")

    request_body = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=request_body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            response_body = (
                response.read()
                .decode("utf-8")
            )

            status_code = response.status

    except urllib.error.HTTPError as exc:

        error_body = (
            exc.read()
            .decode("utf-8")
        )

        print("\nFastAPI returned an error.")

        print("HTTP status:", exc.code)

        print("Response:")
        print(error_body)

        raise

    except urllib.error.URLError as exc:

        raise RuntimeError(
            "Could not connect to FastAPI. "
            "Make sure Uvicorn is running."
        ) from exc

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    result = json.loads(
        response_body
    )

    # --------------------------------------------------------
    # Display response
    # --------------------------------------------------------

    print("\nFastAPI response:")
    print("-" * 70)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    # --------------------------------------------------------
    # Extract prediction
    # --------------------------------------------------------

    api_probability = float(
        result["fraud_probability"]
    )

    api_prediction = int(
        result["fraud_prediction"]
    )

    api_decision = str(
        result["decision"]
    )

    # --------------------------------------------------------
    # Compare with expected MLflow result
    # --------------------------------------------------------

    probability_difference = abs(
        api_probability
        - EXPECTED_MLFLOW_PROBABILITY
    )

    print("\n" + "=" * 70)
    print("MODEL CONSISTENCY CHECK")
    print("=" * 70)

    print(
        "\nExpected MLflow probability:",
        EXPECTED_MLFLOW_PROBABILITY
    )

    print(
        "FastAPI probability:",
        api_probability
    )

    print(
        "Absolute difference:",
        probability_difference
    )

    # --------------------------------------------------------
    # Verify probability
    # --------------------------------------------------------

    if probability_difference <= TOLERANCE:

        print(
            "\nProbability consistency: PASSED"
        )

    else:

        print(
            "\nProbability consistency: FAILED"
        )

        print(
            "\nThe FastAPI prediction does not match "
            "the previously verified MLflow prediction."
        )

        raise AssertionError(
            "FastAPI and MLflow probabilities differ "
            "beyond the allowed tolerance."
        )

    # --------------------------------------------------------
    # Verify prediction
    # --------------------------------------------------------

    if api_prediction != 1:

        raise AssertionError(
            "Expected fraud_prediction=1."
        )

    if api_decision != "FRAUD REVIEW":

        raise AssertionError(
            "Expected decision='FRAUD REVIEW'."
        )

    if int(transaction["isFraud"]) != 1:

        raise AssertionError(
            "Expected actual transaction label to be fraud."
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "Prediction consistency: PASSED"
    )

    print(
        "Decision consistency: PASSED"
    )

    print(
        "Actual fraud label: PASSED"
    )

    print("\n" + "=" * 70)
    print("FASTAPI END-TO-END VALIDATION PASSED")
    print("=" * 70)

    print(
        "\nOffline MLflow result and FastAPI result "
        "are consistent."
    )

    print(
        "\nHTTP status:",
        status_code
    )

    print(
        "Transaction:",
        TRANSACTION_ID
    )

    print(
        "Fraud probability:",
        api_probability
    )

    print(
        "Decision:",
        api_decision
    )

    print("=" * 70)


if __name__ == "__main__":
    main()