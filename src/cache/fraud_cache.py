import json
from typing import Any, Dict, Optional

from src.cache.redis_client import RedisClient


class FraudCache:
    """Cache fraud prediction results in Redis."""

    KEY_PREFIX = "fraud:risk:"
    DEFAULT_TTL_SECONDS = 300

    def __init__(
        self,
        redis_client: RedisClient,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero.")

        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    def _build_key(self, transaction_id: int) -> str:
        """Build the Redis key for a transaction."""
        return f"{self.KEY_PREFIX}{transaction_id}"

    def set_prediction(
        self,
        transaction_id: int,
        prediction: Dict[str, Any],
    ) -> bool:
        """Cache a fraud prediction."""
        key = self._build_key(transaction_id)

        payload = json.dumps(
            prediction,
            separators=(",", ":"),
        )

        return self.redis_client.set(
            key=key,
            value=payload,
            ttl_seconds=self.ttl_seconds,
        )

    def get_prediction(
        self,
        transaction_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached fraud prediction."""
        key = self._build_key(transaction_id)

        cached_value = self.redis_client.get(key)

        if cached_value is None:
            return None

        return json.loads(cached_value)

    def delete_prediction(self, transaction_id: int) -> int:
        """Remove a cached prediction."""
        key = self._build_key(transaction_id)

        return self.redis_client.delete(key)

    def prediction_exists(self, transaction_id: int) -> bool:
        """Check whether a prediction is cached."""
        key = self._build_key(transaction_id)

        return self.redis_client.exists(key)

    def get_ttl(self, transaction_id: int) -> int:
        """Return remaining cache lifetime."""
        key = self._build_key(transaction_id)

        return self.redis_client.ttl(key)