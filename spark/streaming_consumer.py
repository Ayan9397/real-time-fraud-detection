from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fraud-transactions"


TRANSACTION_SCHEMA = StructType(
    [
        StructField("TransactionID", LongType(), True),
        StructField("TransactionDT", LongType(), True),
        StructField("TransactionAmt", DoubleType(), True),
        StructField("ProductCD", StringType(), True),
        StructField("card1", LongType(), True),
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
        StructField("isFraud", LongType(), True),
    ]
)


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("FraudDetectionStreaming")
        .config("spark.sql.shuffle.partitions", "2")
        .config(
            "spark.hadoop.fs.file.impl",
            "org.apache.hadoop.fs.RawLocalFileSystem",
        )
        .config(
            "spark.hadoop.fs.file.impl.disable.cache",
            "true",
        )
        .getOrCreate()
    )


def main() -> None:
    print("=" * 70)
    print("PYSPARK KAFKA STREAMING CONSUMER")
    print("=" * 70)
    print()

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark version: {spark.version}")
    print(f"Kafka broker: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka topic: {KAFKA_TOPIC}")
    print()

    configuration = spark.sparkContext._jsc.hadoopConfiguration()

    print("Filesystem configuration:")
    print(
        "fs.file.impl =",
        configuration.get("fs.file.impl"),
    )
    print(
        "fs.AbstractFileSystem.file.impl =",
        configuration.get("fs.AbstractFileSystem.file.impl"),
    )
    print()

    print("Connecting Spark Structured Streaming to Kafka...")

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            KAFKA_BOOTSTRAP_SERVERS,
        )
        .option(
            "subscribe",
            KAFKA_TOPIC,
        )
        .option(
            "startingOffsets",
            "earliest",
        )
        .option(
            "failOnDataLoss",
            "false",
        )
        .load()
    )

    transactions = (
        kafka_df
        .select(
            col("key").cast("string").alias("kafka_key"),
            col("value").cast("string").alias("json_value"),
            col("partition"),
            col("offset"),
        )
    )

    parsed_transactions = (
        transactions
        .withColumn(
            "transaction",
            from_json(
                col("json_value"),
                TRANSACTION_SCHEMA,
            ),
        )
        .select(
            "kafka_key",
            "partition",
            "offset",
            "transaction.*",
        )
    )

    query = (
        parsed_transactions
        .writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .option("numRows", 20)
        .start()
    )

    print()
    print("Streaming query started successfully.")
    print("Waiting for Kafka transactions...")
    print("Press Ctrl+C to stop.")
    print()

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print()
        print("Stopping Spark streaming query...")
        query.stop()
    finally:
        spark.stop()
        print("Spark session stopped.")


if __name__ == "__main__":
    main()