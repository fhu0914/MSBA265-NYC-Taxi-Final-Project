"""FastAPI +5 optional demo. Run: uvicorn app:app --reload"""
import json
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
import joblib,pandas as pd,numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

app=FastAPI(title="NYC Taxi ETA — MSBA 265 P2",version="1.0",description="Course prototype; not a real-time navigation service.")
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"artifacts"/"taxi_eta_pipeline.joblib"
# Every served prediction is appended here so that src/monitor.py can compare
# live traffic against the training baseline. One JSON object per line.
PREDICTION_LOG=ROOT/"logs"/"predictions.jsonl"

class TaxiRequest(BaseModel):
    distance_km: float = Field(ge=0,le=200,description="Great-circle distance, km; destination must be known")
    pickup_hour: int = Field(ge=0,le=23,description="Local pickup hour, 0–23")

@lru_cache(maxsize=1)
def load_model():
    if not MODEL.exists():
        raise RuntimeError("Model file missing. Run python src/p2_analysis.py first.")
    return joblib.load(MODEL)

def log_prediction(request:TaxiRequest,minutes:float)->None:
    """Append one served request to the monitoring log.

    Logging must never break a prediction that already succeeded, so any
    filesystem problem here is swallowed rather than raised to the caller.
    """
    try:
        PREDICTION_LOG.parent.mkdir(exist_ok=True)
        record={"served_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "distance_km":request.distance_km,"pickup_hour":request.pickup_hour,
                "predicted_minutes":minutes}
        with PREDICTION_LOG.open("a",encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
    except OSError:
        pass

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
    log_prediction(request,round(minutes,2))
    return {"predicted_minutes":round(minutes,2),"model":"distance + pickup hour linear regression",
            "note":"Illustrative estimate; great-circle distance is not road distance."}
