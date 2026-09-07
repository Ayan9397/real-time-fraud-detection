import os
from pathlib import Path
from typing import Optional

import redis
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None


class RedisClient:
    """Redis connection wrapper for the fraud detection platform."""

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        password: Optional[str] = REDIS_PASSWORD,
    ) -> None:
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )

    def ping(self) -> bool:
        """Check whether Redis is reachable."""
        return bool(self.client.ping())

    def set(
        self,
        key: str,
        value: str,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Store a value in Redis, optionally with a TTL."""
        return bool(
            self.client.set(
                name=key,
                value=value,
                ex=ttl_seconds,
            )
        )

    def get(self, key: str) -> Optional[str]:
        """Retrieve a value from Redis."""
        return self.client.get(key)

    def delete(self, key: str) -> int:
        """Delete a Redis key."""
        return int(self.client.delete(key))

    def exists(self, key: str) -> bool:
        """Check whether a Redis key exists."""
        return bool(self.client.exists(key))

    def ttl(self, key: str) -> int:
        """Return remaining TTL in seconds."""
        return int(self.client.ttl(key))

    def close(self) -> None:
        """Close the Redis connection pool."""
        self.client.close()


if __name__ == "__main__":
    redis_client = RedisClient()

    try:
        if redis_client.ping():
            print("Redis connection successful.")
            print(f"Host: {REDIS_HOST}")
            print(f"Port: {REDIS_PORT}")
            print(f"Database: {REDIS_DB}")
    finally:
        redis_client.close()