from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import FraudPrediction, Transaction

class DatabaseService:
def **init**(self, db: Session):
self.db = db

```
def get_transaction(
    self,
    transaction_id: int,
) -> Transaction | None:
    statement = select(Transaction).where(
        Transaction.transaction_id == int(transaction_id)
    )

    return self.db.execute(statement).scalar_one_or_none()

def get_latest_prediction(
    self,
    transaction_id: int,
) -> FraudPrediction | None:
    statement = (
        select(FraudPrediction)
        .where(
            FraudPrediction.transaction_id == int(transaction_id)
        )
        .order_by(FraudPrediction.created_at.desc())
        .limit(1)
    )

    return self.db.execute(statement).scalar_one_or_none()

def save_transaction(
    self,
    transaction_data: dict[str, Any],
) -> Transaction:
    transaction = Transaction(
        transaction_id=int(transaction_data["transaction_id"]),
        transaction_dt=int(transaction_data["transaction_dt"]),
        transaction_amt=Decimal(
            str(transaction_data["transaction_amt"])
        ),
        product_cd=transaction_data.get("product_cd"),
        card1=transaction_data.get("card1"),
        card2=transaction_data.get("card2"),
        card3=transaction_data.get("card3"),
        card4=transaction_data.get("card4"),
        card5=transaction_data.get("card5"),
        card6=transaction_data.get("card6"),
        addr1=transaction_data.get("addr1"),
        addr2=transaction_data.get("addr2"),
        dist1=transaction_data.get("dist1"),
        dist2=transaction_data.get("dist2"),
        p_emaildomain=transaction_data.get("p_emaildomain"),
        r_emaildomain=transaction_data.get("r_emaildomain"),
        actual_fraud=transaction_data.get("actual_fraud"),
    )

    self.db.add(transaction)
    self.db.flush()

    return transaction

def save_prediction(
    self,
    transaction_id: int,
    model_name: str,
    model_version: str,
    fraud_probability: float,
    fraud_prediction: bool,
    decision: str,
    threshold: float,
    prediction_latency_ms: float | None = None,
) -> FraudPrediction:
    prediction = FraudPrediction(
        transaction_id=transaction_id,
        model_name=model_name,
        model_version=model_version,
        fraud_probability=Decimal(
            str(fraud_probability)
        ),
        fraud_prediction=fraud_prediction,
        decision=decision,
        threshold=Decimal(str(threshold)),
        prediction_latency_ms=(
            Decimal(str(prediction_latency_ms))
            if prediction_latency_ms is not None
            else None
        ),
    )

    self.db.add(prediction)
    self.db.flush()

    return prediction

def commit(self) -> None:
    self.db.commit()

def rollback(self) -> None:
    self.db.rollback()
```
