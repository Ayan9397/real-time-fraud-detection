import json
import time
from pathlib import Path

import pandas as pd
import requests
from kafka import KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fraud-transactions"
API_URL = "http://localhost:8000"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "valid.csv"


def create_producer():
    """Create and return a Kafka producer instance."""
    try:
        return KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"Warning: Could not connect to Kafka ({e}). Proceeding with API direct scoring."
        )
        return None


def stream_transactions(num_transactions: int = 25, delay_seconds: float = 0.5):
    """
    Stream real validation transactions into both the Kafka pipeline
    and the real-time FastAPI inference engine.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    print("=" * 70)
    print("LIVE TRANSACTION STREAMING PIPELINE")
    print("=" * 70)
    print(f"Loading {num_transactions} transactions from: {DATASET_PATH.name}")
    print(f"Targeting Kafka topic: {KAFKA_TOPIC} | FastAPI: {API_URL}/predict")
    print("=" * 70)
    print()

    df = pd.read_csv(DATASET_PATH, nrows=num_transactions)
    producer = create_producer()

    sent_count = 0
    fraud_count = 0

    for idx, row in df.iterrows():
        # Convert row to clean dict
        raw_row = row.to_dict()
        transaction = {}
        for k, v in raw_row.items():
            if pd.isna(v):
                transaction[k] = None
            else:
                transaction[k] = v.item() if hasattr(v, "item") else v

        txn_id = int(transaction.get("TransactionID", idx + 1000))
        amt = float(transaction.get("TransactionAmt", 0.0))

        # 1. Produce to Kafka
        kafka_status = "Skipped"
        if producer:
            try:
                future = producer.send(
                    KAFKA_TOPIC,
                    value=transaction,
                    key=str(txn_id).encode("utf-8"),
                )
                future.get(timeout=5)
                kafka_status = "Sent"
            except Exception as exc:  # noqa: BLE001
                kafka_status = f"Err: {exc}"

        # 2. Score with FastAPI /predict endpoint
        api_status = "Skipped"
        decision = "N/A"
        prob = 0.0
        latency_ms = 0.0

        try:
            resp = requests.post(
                f"{API_URL}/predict", json=transaction, timeout=5
            )
            if resp.status_code == 200:
                res = resp.json()
                decision = res.get("decision", "legit")
                prob = res.get("fraud_probability", 0.0)
                latency_ms = res.get("request_latency_ms", 0.0)
                api_status = f"{decision.upper()} ({prob:.3f})"
                if "fraud" in decision.lower():
                    fraud_count += 1
            else:
                api_status = f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            api_status = f"API Err: {exc}"

        sent_count += 1
        tag = "[FRAUD]" if "fraud" in decision.lower() else "[LEGIT]"
        print(
            f"[{sent_count:02d}/{num_transactions}] Txn #{txn_id:<9} "
            f"Amt: ${amt:>7.2f} | Kafka: {kafka_status:<6} | {tag:<7} Model: {api_status:<16} "
            f"({latency_ms:.1f}ms)"
        )

        time.sleep(delay_seconds)

    if producer:
        producer.flush()
        producer.close()

    print()
    print("=" * 70)
    print("STREAMING SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Total Transactions Streamed : {sent_count}")
    print(
        f"Flagged as Fraud            : {fraud_count} ({(fraud_count/sent_count*100):.1f}%)"
    )
    print(f"Flagged as Legitimate       : {sent_count - fraud_count}")
    print()
    print("View real-time graphs in Grafana  : http://localhost:3000")
    print("View live feed in Streamlit       : http://localhost:8501")
    print("=" * 70)


if __name__ == "__main__":
    stream_transactions(num_transactions=20, delay_seconds=0.3)
