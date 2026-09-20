from pathlib import Path
import pickle
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from business_layer import classify_risk
from feature_engineering import feature_engineering

MODEL_PATH = Path(__file__).parent / "model" / "churn_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Telco Customer Churn Prediction API", version="1.0.0")

class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        frame = pd.DataFrame([customer.model_dump()])
        pred = int(model.predict(frame)[0])
        prob = float(model.predict_proba(frame)[0, 1])
        decision = classify_risk(prob)
        return {
            "prediction": "Yes" if pred else "No",
            "churn_probability": round(prob, 4),
            "risk_band": decision.risk_band,
            "recommended_action": decision.recommended_action
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid customer input: {exc}")
