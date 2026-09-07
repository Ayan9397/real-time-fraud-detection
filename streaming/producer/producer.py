import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fraud-transactions"

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "valid.csv"
)


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(
            value
        ).encode("utf-8"),
        acks="all",
        retries=5,
    )


def load_transaction():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print(f"Loading dataset: {DATASET_PATH}")

    df = pd.read_csv(
        DATASET_PATH,
        nrows=1,
    )

    row = df.iloc[0].to_dict()

    transaction = {}

    for key, value in row.items():
        if pd.isna(value):
            transaction[key] = None
        else:
            if hasattr(value, "item"):
                value = value.item()

            transaction[key] = value

    return transaction


def main():
    print("=" * 60)
    print("KAFKA FRAUD TRANSACTION PRODUCER")
    print("=" * 60)

    print()
    print("Connecting to Kafka...")

    producer = create_producer()

    print("Kafka producer connected.")

    transaction = load_transaction()

    transaction_id = int(
        transaction["TransactionID"]
    )

    print()
    print(f"TransactionID: {transaction_id}")
    print(
        f"TransactionAmt: "
        f"{transaction.get('TransactionAmt')}"
    )
    print(
        f"ProductCD: "
        f"{transaction.get('ProductCD')}"
    )

    print()
    print(
        f"Sending transaction to topic: "
        f"{KAFKA_TOPIC}"
    )

    future = producer.send(
        KAFKA_TOPIC,
        value=transaction,
        key=str(transaction_id).encode("utf-8"),
    )

    metadata = future.get(timeout=30)

    producer.flush()
    producer.close()

    print()
    print("=" * 60)
    print("MESSAGE SENT SUCCESSFULLY")
    print("=" * 60)

    print(f"Topic: {metadata.topic}")
    print(f"Partition: {metadata.partition}")
    print(f"Offset: {metadata.offset}")
    print(f"TransactionID: {transaction_id}")

    print()
    print("Kafka producer test PASSED.")


if __name__ == "__main__":
    main()