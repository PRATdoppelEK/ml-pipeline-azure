"""
Prometheus metrics for the ML Pipeline FastAPI service.
Import and use in src/serving/api.py.
"""

import time
from functools import wraps

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


if PROMETHEUS_AVAILABLE:
    # ── Counters ────────────────────────────────────────────────
    PREDICTION_COUNTER = Counter(
        "ml_predictions_total",
        "Total number of predictions made",
        ["model_version", "status"]
    )

    REQUEST_COUNTER = Counter(
        "ml_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"]
    )

    DRIFT_ALERT_COUNTER = Counter(
        "ml_drift_alerts_total",
        "Total data drift alerts triggered"
    )

    # ── Histograms ──────────────────────────────────────────────
    PREDICTION_LATENCY = Histogram(
        "ml_prediction_latency_seconds",
        "Prediction latency in seconds",
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )

    REQUEST_LATENCY = Histogram(
        "ml_request_latency_seconds",
        "HTTP request latency in seconds",
        ["endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    )

    # ── Gauges ──────────────────────────────────────────────────
    MODEL_ACCURACY = Gauge(
        "ml_model_accuracy",
        "Current model accuracy on validation set"
    )

    ACTIVE_REQUESTS = Gauge(
        "ml_active_requests",
        "Number of requests currently being processed"
    )

    DATA_DRIFT_SCORE = Gauge(
        "ml_data_drift_score",
        "Current data drift score (0=no drift, 1=full drift)"
    )


def track_prediction(model_version: str = "1.0.0"):
    """Decorator to track prediction metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return await func(*args, **kwargs)

            ACTIVE_REQUESTS.inc()
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                PREDICTION_COUNTER.labels(
                    model_version=model_version,
                    status="success"
                ).inc()
                return result
            except Exception as e:
                PREDICTION_COUNTER.labels(
                    model_version=model_version,
                    status="error"
                ).inc()
                raise
            finally:
                latency = time.time() - start
                PREDICTION_LATENCY.observe(latency)
                ACTIVE_REQUESTS.dec()
        return wrapper
    return decorator


def get_metrics_response():
    """Return Prometheus metrics in text format."""
    if not PROMETHEUS_AVAILABLE:
        return "# Prometheus client not installed\n# Run: pip install prometheus-client\n"
    return generate_latest().decode("utf-8")


def update_model_metrics(accuracy: float, drift_score: float = 0.0):
    """Update model performance gauges."""
    if PROMETHEUS_AVAILABLE:
        MODEL_ACCURACY.set(accuracy)
        DATA_DRIFT_SCORE.set(drift_score)


if __name__ == "__main__":
    print("Prometheus Metrics Module")
    print(f"Prometheus available: {PROMETHEUS_AVAILABLE}")
    if PROMETHEUS_AVAILABLE:
        PREDICTION_COUNTER.labels(model_version="1.0.0", status="success").inc()
        PREDICTION_COUNTER.labels(model_version="1.0.0", status="success").inc()
        PREDICTION_COUNTER.labels(model_version="1.0.0", status="error").inc()
        MODEL_ACCURACY.set(0.91)
        DATA_DRIFT_SCORE.set(0.12)
        with PREDICTION_LATENCY.time():
            time.sleep(0.05)
        print("\nSample metrics output:")
        print(get_metrics_response()[:500] + "...")
    else:
        print("Install with: pip install prometheus-client")
