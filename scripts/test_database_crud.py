from decimal import Decimal

from sqlalchemy import delete, select

from src.database.connection import SessionLocal
from src.database.models import FraudPrediction, Transaction


TEST_TRANSACTION_ID = 999999999


def main() -> None:
    db = SessionLocal()

    try:
        print("=" * 60)
        print("DATABASE CRUD INTEGRATION TEST")
        print("=" * 60)

        # -------------------------------------------------
        # Clean up any previous test data
        # -------------------------------------------------

        print("\n1. Cleaning previous test data...")

        db.execute(
            delete(FraudPrediction).where(
                FraudPrediction.transaction_id == TEST_TRANSACTION_ID
            )
        )

        db.execute(
            delete(Transaction).where(
                Transaction.transaction_id == TEST_TRANSACTION_ID
            )
        )

        db.commit()

        print("Previous test data removed.")

        # -------------------------------------------------
        # INSERT TRANSACTION
        # -------------------------------------------------

        print("\n2. Inserting test transaction...")

        transaction = Transaction(
            transaction_id=TEST_TRANSACTION_ID,
            transaction_dt=12345678,
            transaction_amt=Decimal("1250.500000"),
            product_cd="W",
            card1=12345,
            card2=456,
            card3=150,
            card4="visa",
            card5=226,
            card6="debit",
            addr1=100,
            addr2=10,
            dist1=5.5,
            dist2=None,
            p_emaildomain="gmail.com",
            r_emaildomain="gmail.com",
            actual_fraud=False,
        )

        db.add(transaction)
        db.commit()
        db.refresh(transaction)

        print("Transaction inserted successfully.")
        print(f"Database row ID: {transaction.id}")
        print(f"Transaction ID: {transaction.transaction_id}")

        # -------------------------------------------------
        # READ TRANSACTION
        # -------------------------------------------------

        print("\n3. Reading transaction...")

        statement = select(Transaction).where(
            Transaction.transaction_id == TEST_TRANSACTION_ID
        )

        stored_transaction = db.scalar(statement)

        if stored_transaction is None:
            raise RuntimeError("Transaction could not be retrieved.")

        print("Transaction retrieved successfully.")
        print(f"Amount: {stored_transaction.transaction_amt}")
        print(f"Product: {stored_transaction.product_cd}")
        print(f"Card type: {stored_transaction.card4}")

        # -------------------------------------------------
        # INSERT FRAUD PREDICTION
        # -------------------------------------------------

        print("\n4. Inserting fraud prediction...")

        prediction = FraudPrediction(
            transaction_id=TEST_TRANSACTION_ID,
            model_name="fraud_detection_xgboost",
            model_version="2",
            fraud_probability=Decimal("0.9854405522"),
            fraud_prediction=True,
            decision="FRAUD REVIEW",
            threshold=Decimal("0.6000"),
            prediction_latency_ms=Decimal("12.3456"),
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        print("Fraud prediction inserted successfully.")
        print(f"Prediction row ID: {prediction.id}")
        print(f"Fraud probability: {prediction.fraud_probability}")
        print(f"Decision: {prediction.decision}")

        # -------------------------------------------------
        # READ FRAUD PREDICTION
        # -------------------------------------------------

        print("\n5. Reading fraud prediction...")

        prediction_statement = (
            select(FraudPrediction)
            .where(
                FraudPrediction.transaction_id == TEST_TRANSACTION_ID
            )
            .order_by(FraudPrediction.created_at.desc())
        )

        stored_prediction = db.scalar(prediction_statement)

        if stored_prediction is None:
            raise RuntimeError(
                "Fraud prediction could not be retrieved."
            )

        print("Fraud prediction retrieved successfully.")
        print(
            f"Probability: "
            f"{stored_prediction.fraud_probability}"
        )
        print(
            f"Prediction: "
            f"{stored_prediction.fraud_prediction}"
        )
        print(
            f"Decision: "
            f"{stored_prediction.decision}"
        )

        # -------------------------------------------------
        # VERIFY FOREIGN KEY RELATIONSHIP
        # -------------------------------------------------

        print("\n6. Verifying transaction relationship...")

        if stored_prediction.transaction_id != stored_transaction.transaction_id:
            raise RuntimeError(
                "Foreign-key relationship verification failed."
            )

        print("Foreign-key relationship verified.")

        # -------------------------------------------------
        # CLEANUP
        # -------------------------------------------------

        print("\n7. Cleaning up test data...")

        db.delete(stored_prediction)
        db.delete(stored_transaction)

        db.commit()

        print("Test data removed successfully.")

        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("DATABASE CRUD INTEGRATION TEST PASSED")
        print("=" * 60)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()