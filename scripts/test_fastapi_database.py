import json
import math
import urllib.request
import urllib.error

import pandas as pd

from src.database.connection import SessionLocal
from src.database.models import FraudPrediction, Transaction


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

PROJECT_ROOT = "."

VALIDATION_FILE = (
    "data/processed/valid.csv"
)

TEST_TRANSACTION_ID = 3400379

EXPECTED_PROBABILITY = 0.9854405522346497

EXPECTED_MODEL = "fraud_detection_xgboost"
EXPECTED_MODEL_VERSION = "2"
EXPECTED_DECISION = "FRAUD REVIEW"
EXPECTED_THRESHOLD = 0.60


# ============================================================
# LOAD REAL TRANSACTION
# ============================================================

def load_transaction() -> tuple[dict, int]:
    print("\n1. Loading real validation transaction...")

    df = pd.read_csv(VALIDATION_FILE)

    print(f"Validation dataset shape: {df.shape}")

    row = df[
        df["TransactionID"] == TEST_TRANSACTION_ID
    ]

    if row.empty:
        raise RuntimeError(
            f"TransactionID {TEST_TRANSACTION_ID} "
            "was not found in validation data."
        )

    row = row.iloc[0]

    actual_label = int(row["isFraud"])

    transaction = row.drop(
        labels=["isFraud"]
    ).to_dict()

    # --------------------------------------------------------
    # Convert pandas NaN values to JSON-compatible None
    # --------------------------------------------------------

    cleaned_transaction = {}

    for key, value in transaction.items():

        if pd.isna(value):
            cleaned_transaction[key] = None

        elif hasattr(value, "item"):
            cleaned_transaction[key] = value.item()

        else:
            cleaned_transaction[key] = value

    print(
        f"TransactionID: "
        f"{TEST_TRANSACTION_ID}"
    )

    print(
        f"Actual fraud label: "
        f"{actual_label}"
    )

    print(
        f"Transaction amount: "
        f"{cleaned_transaction['TransactionAmt']}"
    )

    print(
        f"Features sent: "
        f"{len(cleaned_transaction)}"
    )

    return cleaned_transaction, actual_label


# ============================================================
# CALL FASTAPI
# ============================================================

def call_api(
    transaction: dict,
) -> dict:

    print("\n2. Calling FastAPI /predict...")

    payload = json.dumps(
        transaction
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:

            status_code = response.status

            response_body = response.read().decode(
                "utf-8"
            )

    except urllib.error.HTTPError as exc:

        error_body = exc.read().decode(
            "utf-8"
        )

        raise RuntimeError(
            f"FastAPI returned HTTP {exc.code}:\n"
            f"{error_body}"
        ) from exc

    except urllib.error.URLError as exc:

        raise RuntimeError(
            f"Could not connect to FastAPI:\n"
            f"{exc}"
        ) from exc

    result = json.loads(
        response_body
    )

    print(
        f"HTTP status: "
        f"{status_code}"
    )

    print(
        f"Fraud probability: "
        f"{result['fraud_probability']}"
    )

    print(
        f"Fraud prediction: "
        f"{result['fraud_prediction']}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )

    print(
        f"Database persisted: "
        f"{result['database_persisted']}"
    )

    return result


# ============================================================
# VERIFY API RESPONSE
# ============================================================

def verify_api_response(
    result: dict,
) -> None:

    print("\n3. Verifying API response...")

    if result["model"] != EXPECTED_MODEL:
        raise RuntimeError(
            "Model name mismatch."
        )

    if result["model_version"] != EXPECTED_MODEL_VERSION:
        raise RuntimeError(
            "Model version mismatch."
        )

    probability = float(
        result["fraud_probability"]
    )

    probability_difference = abs(
        probability - EXPECTED_PROBABILITY
    )

    print(
        f"Probability difference: "
        f"{probability_difference}"
    )

    if probability_difference > 1e-5:
        raise RuntimeError(
            "Probability does not match "
            "the previously validated MLflow result."
        )

    if result["fraud_prediction"] != 1:
        raise RuntimeError(
            "Fraud prediction should be 1."
        )

    if result["decision"] != EXPECTED_DECISION:
        raise RuntimeError(
            "Decision mismatch."
        )

    if abs(
        float(result["threshold"])
        - EXPECTED_THRESHOLD
    ) > 1e-9:
        raise RuntimeError(
            "Threshold mismatch."
        )

    if result["database_persisted"] is not True:
        raise RuntimeError(
            "API did not confirm database persistence."
        )

    print(
        "API response verification PASSED."
    )


# ============================================================
# VERIFY POSTGRESQL
# ============================================================

def verify_database() -> None:

    print("\n4. Verifying PostgreSQL records...")

    db = SessionLocal()

    try:

        transaction = db.query(
            Transaction
        ).filter(
            Transaction.transaction_id
            == TEST_TRANSACTION_ID
        ).first()

        if transaction is None:
            raise RuntimeError(
                "Transaction was not found "
                "in PostgreSQL."
            )

        print(
            "Transaction record found."
        )

        print(
            f"Transaction DB ID: "
            f"{transaction.id}"
        )

        print(
            f"Transaction amount: "
            f"{transaction.transaction_amt}"
        )

        prediction = db.query(
            FraudPrediction
        ).filter(
            FraudPrediction.transaction_id
            == TEST_TRANSACTION_ID
        ).order_by(
            FraudPrediction.created_at.desc()
        ).first()

        if prediction is None:
            raise RuntimeError(
                "Fraud prediction was not found "
                "in PostgreSQL."
            )

        print(
            "Fraud prediction record found."
        )

        print(
            f"Model: "
            f"{prediction.model_name}"
        )

        print(
            f"Model version: "
            f"{prediction.model_version}"
        )

        print(
            f"Probability: "
            f"{prediction.fraud_probability}"
        )

        print(
            f"Decision: "
            f"{prediction.decision}"
        )

        print(
            f"Threshold: "
            f"{prediction.threshold}"
        )

        print(
            f"Latency: "
            f"{prediction.prediction_latency_ms} ms"
        )

        # ----------------------------------------------------
        # Verify database values
        # ----------------------------------------------------

        if prediction.model_name != EXPECTED_MODEL:
            raise RuntimeError(
                "Database model name mismatch."
            )

        if prediction.model_version != EXPECTED_MODEL_VERSION:
            raise RuntimeError(
                "Database model version mismatch."
            )

        database_probability = float(
            prediction.fraud_probability
        )

        if abs(
            database_probability
            - EXPECTED_PROBABILITY
        ) > 1e-5:
            raise RuntimeError(
                "Database probability mismatch."
            )

        if prediction.fraud_prediction is not True:
            raise RuntimeError(
                "Database fraud prediction mismatch."
            )

        if prediction.decision != EXPECTED_DECISION:
            raise RuntimeError(
                "Database decision mismatch."
            )

        if abs(
            float(prediction.threshold)
            - EXPECTED_THRESHOLD
        ) > 1e-9:
            raise RuntimeError(
                "Database threshold mismatch."
            )

        if prediction.transaction_id != (
            transaction.transaction_id
        ):
            raise RuntimeError(
                "Foreign-key relationship mismatch."
            )

        print(
            "PostgreSQL verification PASSED."
        )

    finally:
        db.close()


# ============================================================
# CLEANUP
# ============================================================

def cleanup_database() -> None:

    print("\n5. Cleaning up integration-test data...")

    db = SessionLocal()

    try:

        prediction_records = db.query(
            FraudPrediction
        ).filter(
            FraudPrediction.transaction_id
            == TEST_TRANSACTION_ID
        ).all()

        for prediction in prediction_records:
            db.delete(prediction)

        transaction = db.query(
            Transaction
        ).filter(
            Transaction.transaction_id
            == TEST_TRANSACTION_ID
        ).first()

        if transaction is not None:
            db.delete(transaction)

        db.commit()

        print(
            "Integration-test data removed."
        )

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("FASTAPI + MLFLOW + POSTGRESQL END-TO-END TEST")
    print("=" * 70)

    transaction, actual_label = (
        load_transaction()
    )

    result = call_api(
        transaction
    )

    verify_api_response(
        result
    )

    verify_database()

    cleanup_database()

    print("\n" + "=" * 70)
    print(
        "FASTAPI + MLFLOW + POSTGRESQL "
        "END-TO-END VALIDATION PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()