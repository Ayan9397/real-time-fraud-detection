import time

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.schemas.fraud import FraudTransactionRequest
from api.services.database_service import DatabaseService
from api.services.fraud_model import FraudModelService
from src.cache.fraud_cache import FraudCache
from src.cache.redis_client import RedisClient
from src.database.connection import get_db


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "Production-style API for real-time transaction "
        "fraud risk scoring using XGBoost, MLflow, "
        "FastAPI, PostgreSQL, and Redis."
    ),
    version="2.0.0",
)


# ============================================================
# Application Services
# ============================================================

model_service = FraudModelService()

redis_client = RedisClient()

fraud_cache = FraudCache(
    redis_client=redis_client,
    ttl_seconds=300,
)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check(
    db: Session = Depends(get_db),
):
    """
    Check the health of:
    - ML model
    - PostgreSQL
    - Redis
    """

    database_status = "healthy"
    redis_status = "healthy"

    # --------------------------------------------------------
    # PostgreSQL health
    # --------------------------------------------------------

    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database_status = "unhealthy"

    # --------------------------------------------------------
    # Redis health
    # --------------------------------------------------------

    try:
        if not redis_client.ping():
            redis_status = "unhealthy"
    except Exception:
        redis_status = "unhealthy"

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    overall_status = (
        "healthy"
        if database_status == "healthy"
        and redis_status == "healthy"
        else "degraded"
    )

    return {
        "status": overall_status,
        "model": model_service.MODEL_NAME,
        "model_version": model_service.MODEL_VERSION,
        "expected_features": len(
            model_service.expected_features
        ),
        "database": database_status,
        "redis": redis_status,
    }


# ============================================================
# Fraud Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict_fraud(
    transaction: FraudTransactionRequest,
    db: Session = Depends(get_db),
):
    """
    Predict fraud risk for a transaction.

    Request flow:

        Client
          ↓
        FastAPI
          ↓
        Redis cache
          ↓
        PostgreSQL
          ↓
        MLflow / XGBoost
          ↓
        PostgreSQL
          ↓
        Redis
          ↓
        Response
    """

    request_start_time = time.perf_counter()

    transaction_id = transaction.TransactionID

    # --------------------------------------------------------
    # Validate TransactionID
    # --------------------------------------------------------

    if transaction_id is None:
        raise HTTPException(
            status_code=400,
            detail="TransactionID is required.",
        )

    database_service = DatabaseService(db)

    try:
        # ====================================================
        # 1. Redis cache lookup
        # ====================================================

        cached_prediction = fraud_cache.get_prediction(
            transaction_id
        )

        if cached_prediction is not None:
            total_latency_ms = (
                time.perf_counter()
                - request_start_time
            ) * 1000

            return {
                "success": True,
                **cached_prediction,
                "request_latency_ms": round(
                    total_latency_ms,
                    4,
                ),
                "database_persisted": True,
                "cache_hit": True,
                "database_hit": False,
            }

        # ====================================================
        # 2. PostgreSQL lookup
        # ====================================================

        existing_transaction = (
            database_service.get_transaction(
                transaction_id
            )
        )

        if existing_transaction is not None:
            existing_prediction = (
                database_service.get_latest_prediction(
                    transaction_id
                )
            )

            if existing_prediction is not None:
                prediction_latency_ms = (
                    float(
                        existing_prediction.prediction_latency_ms
                    )
                    if existing_prediction.prediction_latency_ms
                    is not None
                    else 0.0
                )

                database_result = {
                    "model": (
                        existing_prediction.model_name
                    ),
                    "model_version": (
                        existing_prediction.model_version
                    ),
                    "transaction_id": transaction_id,
                    "fraud_probability": float(
                        existing_prediction.fraud_probability
                    ),
                    "fraud_prediction": bool(
                        existing_prediction.fraud_prediction
                    ),
                    "decision": (
                        existing_prediction.decision
                    ),
                    "threshold": float(
                        existing_prediction.threshold
                    ),
                    "prediction_latency_ms": (
                        prediction_latency_ms
                    ),
                }

                # ------------------------------------------------
                # Restore PostgreSQL result into Redis cache.
                # ------------------------------------------------

                fraud_cache.set_prediction(
                    transaction_id=transaction_id,
                    prediction=database_result,
                )

                total_latency_ms = (
                    time.perf_counter()
                    - request_start_time
                ) * 1000

                return {
                    "success": True,
                    **database_result,
                    "request_latency_ms": round(
                        total_latency_ms,
                        4,
                    ),
                    "database_persisted": True,
                    "cache_hit": False,
                    "database_hit": True,
                }

        # ====================================================
        # 3. ML model inference
        # ====================================================

        transaction_data = (
            transaction.to_transaction_dict()
        )

        inference_start_time = time.perf_counter()

        result = model_service.predict(
            transaction_data
        )

        prediction_latency_ms = (
            time.perf_counter()
            - inference_start_time
        ) * 1000

        # ====================================================
        # 4. Prepare PostgreSQL transaction record
        # ====================================================

        database_transaction_data = {
            "transaction_id": transaction_id,
            "transaction_dt": int(
                transaction.TransactionDT
            ),
            "transaction_amt": float(
                transaction.TransactionAmt
            ),
            "product_cd": transaction.ProductCD,

            "card1": (
                int(transaction.card1)
                if transaction.card1 is not None
                else None
            ),

            "card2": (
                int(transaction.card2)
                if transaction.card2 is not None
                else None
            ),

            "card3": (
                int(transaction.card3)
                if transaction.card3 is not None
                else None
            ),

            "card4": transaction.card4,

            "card5": (
                int(transaction.card5)
                if transaction.card5 is not None
                else None
            ),

            "card6": transaction.card6,

            "addr1": (
                int(transaction.addr1)
                if transaction.addr1 is not None
                else None
            ),

            "addr2": (
                int(transaction.addr2)
                if transaction.addr2 is not None
                else None
            ),

            "dist1": transaction.dist1,
            "dist2": transaction.dist2,

            "p_emaildomain": (
                transaction.P_emaildomain
            ),

            "r_emaildomain": (
                transaction.R_emaildomain
            ),

            "actual_fraud": None,
        }

        # ====================================================
        # 5. Persist transaction
        # ====================================================

        database_service.save_transaction(
            database_transaction_data
        )

        # ====================================================
        # 6. Persist prediction
        # ====================================================

        threshold = model_service.THRESHOLD

        database_service.save_prediction(
            transaction_id=transaction_id,
            model_name=model_service.MODEL_NAME,
            model_version=model_service.MODEL_VERSION,
            fraud_probability=(
                result["fraud_probability"]
            ),
            fraud_prediction=bool(
                result["fraud_prediction"]
            ),
            decision=result["decision"],
            threshold=threshold,
            prediction_latency_ms=(
                prediction_latency_ms
            ),
        )

        # ====================================================
        # 7. Commit database transaction
        # ====================================================

        database_service.commit()

        # ====================================================
        # 8. Prepare Redis cache result
        # ====================================================

        cache_result = {
            "model": model_service.MODEL_NAME,
            "model_version": (
                model_service.MODEL_VERSION
            ),
            "transaction_id": transaction_id,
            "fraud_probability": (
                result["fraud_probability"]
            ),
            "fraud_prediction": (
                result["fraud_prediction"]
            ),
            "decision": result["decision"],
            "threshold": threshold,
            "prediction_latency_ms": round(
                prediction_latency_ms,
                4,
            ),
        }

        # ====================================================
        # 9. Write prediction to Redis
        # ====================================================

        cache_written = fraud_cache.set_prediction(
            transaction_id=transaction_id,
            prediction=cache_result,
        )

        # ====================================================
        # 10. Calculate complete request latency
        # ====================================================

        total_latency_ms = (
            time.perf_counter()
            - request_start_time
        ) * 1000

        # ====================================================
        # 11. Return API response
        # ====================================================

        return {
            "success": True,
            **cache_result,
            "request_latency_ms": round(
                total_latency_ms,
                4,
            ),
            "database_persisted": True,
            "cache_hit": False,
            "database_hit": False,
            "cache_written": cache_written,
        }

    # ========================================================
    # HTTP errors
    # ========================================================

    except HTTPException:
        db.rollback()
        raise

    # ========================================================
    # Unexpected errors
    # ========================================================

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
