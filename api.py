"""
FastAPI service for Loan Default Prediction.
Run with: uvicorn api:app --reload --port 8000
Docs at: http://localhost:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

from train import load_data, build_model

app = FastAPI(title="Loan Default Prediction API", version="1.0")

_model = None
_scaler = None
_feature_names = None


class Applicant(BaseModel):
    income: float = Field(..., example=55000, ge=0)
    credit_score: float = Field(..., example=650, ge=300, le=850)
    debt_to_income: float = Field(..., example=0.3, ge=0, le=1)
    loan_amount: float = Field(..., example=20000, ge=0)
    employment_years: int = Field(..., example=5, ge=0)
    num_late_payments: int = Field(..., example=0, ge=0)


class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier: str


@app.on_event("startup")
def train_on_startup():
    """Trains once when the API starts. In production, replace this with
    loading a pre-trained model artifact from disk / a model registry."""
    global _model, _scaler, _feature_names
    df = load_data()
    X = df.drop(columns="default")
    y = df["default"]
    _scaler = StandardScaler().fit(X)
    _model = build_model(X.shape[1])
    _model.fit(_scaler.transform(X), y, epochs=15, batch_size=64, verbose=0)
    _feature_names = list(X.columns)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: Applicant):
    import pandas as pd
    row = pd.DataFrame([applicant.dict()])[_feature_names]
    scaled = _scaler.transform(row)
    prob = float(_model.predict(scaled, verbose=0)[0][0])

    if prob > 0.5:
        tier = "high_risk"
    elif prob > 0.25:
        tier = "moderate_risk"
    else:
        tier = "low_risk"

    return PredictionResponse(default_probability=round(prob, 4), risk_tier=tier)
