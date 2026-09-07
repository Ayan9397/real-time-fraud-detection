from src.cache.fraud_cache import FraudCache
from src.cache.redis_client import RedisClient


TEST_TRANSACTION_ID = 999999997


def main() -> None:
    redis_client = RedisClient()
    fraud_cache = FraudCache(
        redis_client=redis_client,
        ttl_seconds=60,
    )

    try:
        print("=" * 60)
        print("REDIS INTEGRATION TEST")
        print("=" * 60)

        print("\n1. Testing Redis connection...")
        assert redis_client.ping() is True
        print("   PASS: Redis connection successful.")

        prediction = {
            "transaction_id": TEST_TRANSACTION_ID,
            "fraud_probability": 0.9854405522,
            "fraud_prediction": 1,
            "decision": "FRAUD REVIEW",
            "model_name": "fraud_detection_xgboost",
            "model_version": "2",
            "threshold": 0.60,
        }

        print("\n2. Writing prediction to Redis...")
        success = fraud_cache.set_prediction(
            transaction_id=TEST_TRANSACTION_ID,
            prediction=prediction,
        )

        assert success is True
        print("   PASS: Prediction cached.")

        print("\n3. Checking cache existence...")
        assert fraud_cache.prediction_exists(TEST_TRANSACTION_ID)
        print("   PASS: Cache key exists.")

        print("\n4. Reading prediction from Redis...")
        cached_prediction = fraud_cache.get_prediction(
            TEST_TRANSACTION_ID
        )

        assert cached_prediction is not None
        assert (
            cached_prediction["fraud_probability"]
            == prediction["fraud_probability"]
        )
        assert (
            cached_prediction["decision"]
            == prediction["decision"]
        )

        print("   PASS: Cached prediction matches original.")

        print("\n5. Checking TTL...")
        ttl = fraud_cache.get_ttl(TEST_TRANSACTION_ID)

        assert 0 < ttl <= 60
        print(f"   PASS: TTL = {ttl} seconds.")

        print("\n6. Deleting cache entry...")
        deleted = fraud_cache.delete_prediction(
            TEST_TRANSACTION_ID
        )

        assert deleted == 1
        print("   PASS: Cache entry deleted.")

        print("\n7. Confirming deletion...")
        assert not fraud_cache.prediction_exists(
            TEST_TRANSACTION_ID
        )
        print("   PASS: Cache entry no longer exists.")

        print("\n" + "=" * 60)
        print("REDIS INTEGRATION TEST PASSED")
        print("=" * 60)

    finally:
        redis_client.close()


if __name__ == "__main__":
    main()