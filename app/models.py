# app/models.py

import joblib
from pathlib import Path

MODEL_PATH = Path("models/diabetes_ridge.joblib")

def load_model():
    """Load the persisted trained model (LinearRegression, Ridge, etc)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()
model_version = "v1.0"
