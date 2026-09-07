import json
from kafka import KafkaConsumer


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fraud-transactions"
KAFKA_GROUP_ID = "fraud-detection-consumer"


def create_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )


def main():
    print("=" * 60)
    print("KAFKA FRAUD TRANSACTION CONSUMER")
    print("=" * 60)

    print()
    print("Connecting to Kafka...")

    consumer = create_consumer()

    print("Kafka consumer connected.")
    print(f"Listening to topic: {KAFKA_TOPIC}")
    print()
    print("Waiting for transactions...")
    print("Press Ctrl+C to stop.")
    print()

    try:
        for message in consumer:
            transaction = message.value

            transaction_id = transaction.get(
                "TransactionID"
            )

            transaction_amt = transaction.get(
                "TransactionAmt"
            )

            product_cd = transaction.get(
                "ProductCD"
            )

            print("=" * 60)
            print("TRANSACTION RECEIVED")
            print("=" * 60)

            print(f"Topic: {message.topic}")
            print(f"Partition: {message.partition}")
            print(f"Offset: {message.offset}")
            print(f"TransactionID: {transaction_id}")
            print(f"TransactionAmt: {transaction_amt}")
            print(f"ProductCD: {product_cd}")

            print()
            print("Consumer successfully received Kafka message.")
            print()

    except KeyboardInterrupt:
        print()
        print("Stopping Kafka consumer...")

    finally:
        consumer.close()
        print("Kafka consumer closed.")


if __name__ == "__main__":
    main()