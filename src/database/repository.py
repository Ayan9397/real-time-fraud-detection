from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import FraudPrediction, Transaction


# ---------------------------------------------------------
# Transaction operations
# ---------------------------------------------------------

def create_transaction(
    db: Session,
    transaction_data: dict,
) -> Transaction:
    """
    Insert a transaction into PostgreSQL.
    """

    transaction = Transaction(**transaction_data)

    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


def get_transaction(
    db: Session,
    transaction_id: int,
) -> Transaction | None:
    """
    Retrieve a transaction by TransactionID.
    """

    statement = select(Transaction).where(
        Transaction.transaction_id == transaction_id
    )

    return db.scalar(statement)


# ---------------------------------------------------------
# Fraud prediction operations
# ---------------------------------------------------------

def create_prediction(
    db: Session,
    prediction_data: dict,
) -> FraudPrediction:
    """
    Store a fraud prediction in PostgreSQL.
    """

    prediction = FraudPrediction(**prediction_data)

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return prediction


def get_predictions_for_transaction(
    db: Session,
    transaction_id: int,
) -> list[FraudPrediction]:
    """
    Retrieve all predictions associated with a transaction.
    """

    statement = (
        select(FraudPrediction)
        .where(FraudPrediction.transaction_id == transaction_id)
        .order_by(FraudPrediction.created_at.desc())
    )

    return list(db.scalars(statement).all())


# ---------------------------------------------------------
# Recent predictions
# ---------------------------------------------------------

def get_recent_predictions(
    db: Session,
    limit: int = 20,
) -> list[FraudPrediction]:
    """
    Retrieve the most recent fraud predictions.
    """

    statement = (
        select(FraudPrediction)
        .order_by(FraudPrediction.created_at.desc())
        .limit(limit)
    )

    return list(db.scalars(statement).all())