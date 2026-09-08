from decimal import Decimal

from api.services.database_service import DatabaseService


class FakeSession:
    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.added = []
        self.flushed = False

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        self.flushed = True

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_save_transaction():
    db = FakeSession()
    service = DatabaseService(db)

    transaction_data = {
        "transaction_id": 999999998,
        "transaction_dt": 123456,
        "transaction_amt": 1265.50,
        "product_cd": "W",
        "card1": 12345,
        "card2": 111,
        "card3": 150,
        "card4": "visa",
        "card5": 226,
        "card6": "credit",
        "addr1": 100,
        "addr2": 87,
        "dist1": 10.5,
        "dist2": 20.5,
        "p_emaildomain": "gmail.com",
        "r_emaildomain": "gmail.com",
        "actual_fraud": False,
    }

    transaction = service.save_transaction(transaction_data)

    assert transaction.transaction_id == 999999998
    assert transaction.transaction_dt == 123456
    assert transaction.transaction_amt == Decimal("1265.5")
    assert transaction.product_cd == "W"
    assert transaction.actual_fraud is False
    assert len(db.added) == 1
    assert db.flushed is True


def test_save_prediction():
    db = FakeSession()
    service = DatabaseService(db)

    prediction = service.save_prediction(
        transaction_id=999999998,
        model_name="fraud_detection_xgboost",
        model_version="4",
        fraud_probability=0.043047092854976654,
        fraud_prediction=False,
        decision="LEGITIMATE",
        threshold=0.60,
        prediction_latency_ms=854.4269,
    )

    assert prediction.transaction_id == 999999998
    assert prediction.model_name == "fraud_detection_xgboost"
    assert prediction.model_version == "4"
    assert prediction.fraud_probability == Decimal(
        "0.043047092854976654"
    )
    assert prediction.fraud_prediction is False
    assert prediction.decision == "LEGITIMATE"
    assert prediction.threshold == Decimal("0.6")
    assert prediction.prediction_latency_ms == Decimal("854.4269")
    assert len(db.added) == 1
    assert db.flushed is True


def test_commit():
    db = FakeSession()
    service = DatabaseService(db)

    service.commit()

    assert db.committed is True


def test_rollback():
    db = FakeSession()
    service = DatabaseService(db)

    service.rollback()

    assert db.rolled_back is True
