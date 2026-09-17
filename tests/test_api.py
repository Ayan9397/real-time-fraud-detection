from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] in {"healthy", "degraded"}
    assert data["model"] == "fraud_detection_xgboost"
    assert data["model_version"] == "6"
    assert data["expected_features"] == 864
    assert "database" in data
    assert "redis" in data


def test_predict_validation_error():
    response = client.post(
        "/predict",
        json={
            "TransactionID": 999999997,
            "TransactionAmt": 100.0,
        },
    )

    assert response.status_code == 422


def test_predict_missing_transaction_id():
    response = client.post(
        "/predict",
        json={
            "TransactionDT": 123456.0,
            "TransactionAmt": 100.0,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "TransactionID is required."


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    content = response.text
    assert "http_requests_total" in content
    assert "fraud_predictions_total" in content
    assert "fraud_risk_score" in content
