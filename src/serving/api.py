"""
REST API serving for trained ML models (FastAPI).
Author: Prateek Gaur
"""

import os
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── State ──────────────────────────────────────────────────────
_model = None

def load_model(model_path: str):
    global _model
    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")

# ── Lifespan (replaces on_event which is deprecated) ───────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    if os.path.exists(model_path):
        load_model(model_path)
        logger.info(f"Model loaded at startup: {model_path}")
    else:
        logger.warning(f"Model not found at: {model_path}")
    yield

app = FastAPI(
    title="ML Pipeline Prediction API",
    description="Prateek Gaur — End-to-End ML Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Schemas ────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature name → value mapping")

class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dicts")

class PredictionResponse(BaseModel):
    prediction: Any
    probability: Optional[float] = None
    model_version: str = "1.0.0"

class BatchPredictionResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[float]] = None
    count: int

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/model/info")
def model_info():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    model_step = _model.named_steps.get("model", _model)
    info = {
        "type": type(model_step).__name__,
        "params": model_step.get_params(),
    }
    if hasattr(model_step, "feature_importances_"):
        info["feature_importances"] = model_step.feature_importances_.tolist()
    return info

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    try:
        from monitoring.prometheus_metrics import get_metrics_response
        return get_metrics_response()
    except Exception:
        return "# Prometheus metrics not available\n"

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        df   = pd.DataFrame([request.features])
        pred = _model.predict(df)[0]
        pred = int(pred) if hasattr(pred, "item") else pred
        prob = None
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(df)[0]
            prob  = float(max(proba))
        return PredictionResponse(prediction=pred, probability=prob)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        df    = pd.DataFrame(request.records)
        preds = [int(p) if hasattr(p, "item") else p for p in _model.predict(df)]
        probs = None
        if hasattr(_model, "predict_proba"):
            probs = _model.predict_proba(df).max(axis=1).tolist()
        return BatchPredictionResponse(predictions=preds, probabilities=probs, count=len(preds))
    except Exception as e:
        raise HTTPException(400, str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
