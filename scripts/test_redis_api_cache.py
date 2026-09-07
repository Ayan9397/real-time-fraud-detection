import sys
import json
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = "http://127.0.0.1:8000"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "valid.csv"

FIRST_REQUEST_TIMEOUT = 120
SECOND_REQUEST_TIMEOUT = 30


def load_test_transaction():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {DATASET_PATH}"
        )

    print(f"Loading dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, nrows=1)

    if "TransactionID" not in df.columns:
        raise RuntimeError(
            "TransactionID column not found in validation dataset."
        )

    row = df.iloc[0].to_dict()

    transaction_id = int(row["TransactionID"])

    cleaned = {}

    for key, value in row.items():
        if pd.isna(value):
            cleaned[key] = None
        else:
            if hasattr(value, "item"):
                value = value.item()
            cleaned[key] = value

    print(f"Selected TransactionID: {transaction_id}")

    return cleaned


def call_api(transaction, timeout):
    start = time.perf_counter()

    response = requests.post(
        f"{API_URL}/predict",
        json=transaction,
        timeout=timeout,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    return response.json(), elapsed_ms


def main():
    print()
    print("=" * 60)
    print("REDIS + FASTAPI CACHE INTEGRATION TEST")
    print("=" * 60)

    print()
    print("[1/5] Checking API health...")

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=30,
    )

    print(f"HTTP status: {health_response.status_code}")
    print(health_response.text)

    health_response.raise_for_status()

    health = health_response.json()

    if health.get("redis") != "healthy":
        raise RuntimeError(f"Redis is not healthy: {health}")

    if health.get("database") != "healthy":
        raise RuntimeError(f"Database is not healthy: {health}")

    print("Health check: PASS")

    print()
    print("[2/5] Loading test transaction...")

    transaction = load_test_transaction()

    transaction_id = int(transaction["TransactionID"])

    print(f"TransactionID: {transaction_id}")
    print(f"TransactionAmt: {transaction.get('TransactionAmt')}")
    print(f"ProductCD: {transaction.get('ProductCD')}")

    print()
    print("[3/5] Sending FIRST request...")

    first_result, first_elapsed = call_api(
        transaction,
        FIRST_REQUEST_TIMEOUT,
    )

    print()
    print("=" * 60)
    print("FIRST REQUEST")
    print("=" * 60)
    print(json.dumps(first_result, indent=2))
    print(f"Measured request time: {first_elapsed:.3f} ms")
    print(f"Cache hit: {first_result.get('cache_hit')}")
    print(f"Database hit: {first_result.get('database_hit')}")
    print(f"Cache written: {first_result.get('cache_written')}")

    print()
    print("[4/5] Sending SECOND request...")

    second_result, second_elapsed = call_api(
        transaction,
        SECOND_REQUEST_TIMEOUT,
    )

    print()
    print("=" * 60)
    print("SECOND REQUEST")
    print("=" * 60)
    print(json.dumps(second_result, indent=2))
    print(f"Measured request time: {second_elapsed:.3f} ms")
    print(f"Cache hit: {second_result.get('cache_hit')}")
    print(f"Database hit: {second_result.get('database_hit')}")
    print(f"Cache written: {second_result.get('cache_written')}")

    print()
    print("[5/5] Validating cache behavior...")

    first_probability = float(
        first_result["fraud_probability"]
    )

    second_probability = float(
        second_result["fraud_probability"]
    )

    difference = abs(
        first_probability - second_probability
    )

    print(f"First probability:  {first_probability:.10f}")
    print(f"Second probability: {second_probability:.10f}")
    print(f"Probability difference: {difference:.12f}")

    if difference > 1e-6:
        raise AssertionError(
            "Fraud probabilities are inconsistent."
        )

    if first_result.get("success") is not True:
        raise AssertionError(
            "First API request failed."
        )

    if second_result.get("success") is not True:
        raise AssertionError(
            "Second API request failed."
        )

    if second_result.get("cache_hit") is not True:
        raise AssertionError(
            "Second request was not served from Redis cache."
        )

    print()
    print("=" * 60)
    print("CACHE TEST PASSED")
    print("=" * 60)

    print(f"First request:  {first_elapsed:.3f} ms")
    print(f"Second request: {second_elapsed:.3f} ms")

    if second_elapsed > 0:
        speedup = first_elapsed / second_elapsed
        print(f"Observed speedup: {speedup:.1f}x")

    print()
    print("Redis successfully served the second request from cache.")


if __name__ == "__main__":
    main()