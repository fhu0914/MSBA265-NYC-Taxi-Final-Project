"""Independent artifact reload and predict test: python src/check_saved_model.py"""
from pathlib import Path
import joblib,pandas as pd,numpy as np
p=Path(__file__).resolve().parents[1]/"artifacts"/"taxi_eta_pipeline.joblib"
a=joblib.load(p)
x=pd.DataFrame([{"distance_km":5.0,"pickup_hour":14}])[a["features"]]
minutes=float(np.clip(np.maximum(np.expm1(a["model"].predict(x)),0)[0]/60,0,360))
assert np.isfinite(minutes) and minutes>=0
print(f"PASS: reloaded model prediction 5km, 14:00 -> {minutes:.2f} minutes")
