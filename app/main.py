# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import joblib
import time
from pathlib import Path

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
MODEL_PATH = Path("models/diabetes_ridge.joblib")
model = joblib.load(MODEL_PATH)

MODEL_VERSION = "v1.0"

# ---------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    ["endpoint"]
)

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(title="Regression Model Server")


# ---------------------------------------------------------
# Request Schema
# ---------------------------------------------------------
class PredictRequest(BaseModel):
    x1: float
    x2: float


class PredictResponse(BaseModel):
    score: float
    model_version: str


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    # Convert to 2D array
    X = [[payload.x1, payload.x2]]
    score = float(model.predict(X)[0])

    duration = time.time() - start
    REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)

    return {"score": score, "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    REQUEST_COUNT.labels(endpoint="/metrics").inc()
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
