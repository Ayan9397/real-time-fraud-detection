from prometheus_client import Counter, Histogram

# ============================================================
# HTTP Request Metrics
# ============================================================

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total count of HTTP requests processed by endpoint and status.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ============================================================
# Business & ML Inference Metrics
# ============================================================

FRAUD_PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total count of fraud model predictions categorized by decision.",
    ["decision", "model_version"],
)

FRAUD_RISK_SCORE = Histogram(
    "fraud_risk_score",
    "Distribution of fraud probability scores (0.0 to 1.0).",
    buckets=(
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        1.0,
    ),
)

FRAUD_INFERENCE_DURATION_SECONDS = Histogram(
    "fraud_inference_duration_seconds",
    "Latency of XGBoost ML inference alone in seconds.",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# ============================================================
# Cache Observability Metrics
# ============================================================

CACHE_REQUESTS_TOTAL = Counter(
    "fraud_cache_requests_total",
    "Total Redis cache prediction lookups categorized by hit or miss.",
    ["result"],
)

