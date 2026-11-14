"""FastAPI +5 optional demo. Run: uvicorn app:app --reload"""
from pathlib import Path
from functools import lru_cache
import joblib,pandas as pd,numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

app=FastAPI(title="NYC Taxi ETA — MSBA 265 P2",version="1.0",description="Course prototype; not a real-time navigation service.")
MODEL=Path(__file__).resolve().parent/"artifacts"/"taxi_eta_pipeline.joblib"

class TaxiRequest(BaseModel):
    distance_km: float = Field(ge=0,le=200,description="Great-circle distance, km; destination must be known")
    pickup_hour: int = Field(ge=0,le=23,description="Local pickup hour, 0–23")

@lru_cache(maxsize=1)
def load_model():
    if not MODEL.exists():
        raise RuntimeError("Model file missing. Run python src/p2_analysis.py first.")
    return joblib.load(MODEL)

@app.get("/health")
def health():
    return {"status":"ok","model_available":MODEL.is_file()}

@app.post("/predict")
def predict(request:TaxiRequest):
    try:
        bundle=load_model()
    except Exception as e:
        raise HTTPException(status_code=503,detail="Model unavailable; run training step.") from e
    x=pd.DataFrame([request.model_dump()])[bundle["features"]]
    pred_log=float(bundle["model"].predict(x)[0])
    minutes=float(np.clip(max(np.expm1(np.clip(pred_log,-10,30)),0)/60,0,360))
    return {"predicted_minutes":round(minutes,2),"model":"distance + pickup hour linear regression",
            "note":"Illustrative estimate; great-circle distance is not road distance."}
