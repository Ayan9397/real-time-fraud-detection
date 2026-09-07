from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    DoubleType,
    LongType,
)


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fraud-transactions"
CHECKPOINT_LOCATION = "/tmp/fraud-detection-checkpoint"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("FraudDetectionStreamingProcessor")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def create_transaction_schema():
    return StructType(
        [
            StructField("TransactionID", LongType(), True),
            StructField("TransactionDT", LongType(), True),
            StructField("TransactionAmt", DoubleType(), True),
            StructField("ProductCD", StringType(), True),
            StructField("card1", DoubleType(), True),
            StructField("card2", DoubleType(), True),
            StructField("card3", DoubleType(), True),
            StructField("card4", StringType(), True),
            StructField("card5", DoubleType(), True),
            StructField("card6", StringType(), True),
            StructField("addr1", DoubleType(), True),
            StructField("addr2", DoubleType(), True),
            StructField("dist1", DoubleType(), True),
            StructField("dist2", DoubleType(), True),
            StructField("P_emaildomain", StringType(), True),
            StructField("R_emaildomain", StringType(), True),
            StructField("isFraud", DoubleType(), True),
        ]
    )


def main():
    print("=" * 70)
    print("REAL-TIME FRAUD DETECTION - SPARK STREAM PROCESSOR")
    print("=" * 70)
    print()

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("Spark session created.")
    print(f"Kafka broker: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka topic: {KAFKA_TOPIC}")
    print()

    schema = create_transaction_schema()

    print("Connecting to Kafka...")

    kafka_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    transactions = (
        kafka_stream
        .selectExpr(
            "CAST(key AS STRING) AS message_key",
            "CAST(value AS STRING) AS message_value",
            "topic",
            "partition",
            "offset",
        )
    )

    parsed_transactions = (
        transactions
        .select(
            "message_key",
            "topic",
            "partition",
            "offset",
            from_json(
                col("message_value"),
                schema,
            ).alias("transaction"),
        )
        .select(
            "message_key",
            "topic",
            "partition",
            "offset",
            "transaction.*",
        )
    )

    processed_transactions = parsed_transactions.select(
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
        "isFraud",
        "topic",
        "partition",
        "offset",
    )

    print("Kafka stream connected.")
    print()
    print("Starting streaming query...")
    print("Waiting for Kafka transactions...")
    print("Press Ctrl+C to stop.")
    print()

    query = (
        processed_transactions
        .writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .option("numRows", 20)
        .option(
            "checkpointLocation",
            CHECKPOINT_LOCATION,
        )
        .start()
    )

    try:
        query.awaitTermination()

    except KeyboardInterrupt:
        print()
        print("Stopping streaming query...")
        query.stop()

    finally:
        spark.stop()
        print("Spark session stopped.")


if __name__ == "__main__":
    main()
