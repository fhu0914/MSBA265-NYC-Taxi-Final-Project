"""P2: reproducible baseline + two regression models and held-out evaluation.

Run from the project root: python src/p2_analysis.py
Data: data/train.zip, downloaded separately from Kaggle by each authorized reviewer.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "train.zip"
ARTIFACT = ROOT / "artifacts" / "taxi_eta_pipeline.joblib"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
ARTIFACT.parent.mkdir(exist_ok=True)
SEED = 42


def haversine_km(df: pd.DataFrame) -> np.ndarray:
    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"].to_numpy() - df["pickup_longitude"].to_numpy())
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def predict_minutes(model, x: pd.DataFrame) -> np.ndarray:
    # Model predicts log1p(seconds); cap 0–6 hours for defined urban-trip scope.
    log_seconds = model.predict(x)
    return np.clip(np.maximum(np.expm1(np.clip(log_seconds, -10, 30)), 0) / 60, 0, 360)


def stats(y_min, pred_min):
    y=np.asarray(y_min,float); p=np.asarray(pred_min,float)
    return {
        "rmsle":round(float(np.sqrt(np.mean((np.log1p(y*60)-np.log1p(p*60))**2))),4),
        "mae_min":round(float(np.mean(np.abs(y-p))),3),
        "within_3min_pct":round(float(100*np.mean(np.abs(y-p)<=3)),2),
    }

if not DATA.exists():
    raise SystemExit(f"Missing {DATA}. Read README.md to download Kaggle train.zip (do not unzip).")

train=pd.read_csv(DATA, usecols=["pickup_datetime","pickup_latitude","pickup_longitude", "dropoff_latitude","dropoff_longitude","trip_duration"])
original_n=len(train)
train=train.dropna().copy()
train=train[train.trip_duration>0].copy()
train["pickup_hour"]=pd.to_datetime(train.pickup_datetime,errors="coerce").dt.hour
train=train.dropna(subset=["pickup_hour"])
train["pickup_hour"]=train["pickup_hour"].astype(int)
train["distance_km"]=haversine_km(train)
train=train[np.isfinite(train.distance_km)].copy()
# Explicit scope for reliable trip estimates: remove implausible GPS jumps
# and very long trips (>6h) that are not representative of ordinary urban rides.
# We evaluate only this defined subset; we do NOT claim performance on outliers.
pre_filter_n=len(train)
train=train[(train.distance_km<=200)&(train.trip_duration<=6*3600)].copy()
excluded_n=pre_filter_n-len(train)
valid_n=len(train)
print(f"Rows loaded: {original_n:,}; rows modeled: {valid_n:,}; excluded outliers: {excluded_n:,}")

# Preserve P1's fixed 80/20 split for comparable scores.
train_set,val_set=train_test_split(train,test_size=.20,random_state=SEED)
ytr_log=np.log1p(train_set.trip_duration.to_numpy())
yval_min=val_set.trip_duration.to_numpy()/60
baseline_sec=float(train_set.trip_duration.median())
baseline_pred=np.repeat(baseline_sec/60,len(val_set))

results=[]
preds={}
s=stats(yval_min,baseline_pred)
results.append({"model":"Median baseline",**s})
preds["Median baseline"]=baseline_pred

models={}
for name, cols in [
    ("Distance-only linear regression",["distance_km"]),
    ("Distance + pickup hour regression",["distance_km","pickup_hour"]),
]:
    model=LinearRegression()
    model.fit(train_set[cols],ytr_log)
    pred=predict_minutes(model,val_set[cols])
    results.append({"model":name,**stats(yval_min,pred)})
    preds[name]=pred
    models[name]=(model,cols)

m,features=models["Distance + pickup hour regression"]
joblib.dump({"model":m,"features":features,"response_unit":"minutes","seed":SEED,
    "target_transform":"log1p(seconds)","training_rows":len(train_set),
    "note":"Predictors are great-circle distance (km) and pickup hour; destination assumed known."},ARTIFACT)

# Fresh process style reload test: we actually load the saved artifact here.
loaded=joblib.load(ARTIFACT)
example=pd.DataFrame([{"distance_km":5.0,"pickup_hour":14}])[loaded["features"]]
example_output=round(float(predict_minutes(loaded["model"],example)[0]),2)
assert np.isfinite(example_output) and example_output>=0

metrics=pd.DataFrame(results)
metrics.to_csv(REPORTS/"model_comparison.csv",index=False)
with open(REPORTS/"summary.json","w",encoding="utf-8") as f:
    json.dump({"data_rows_total":original_n,"rows_modeled":valid_n,
         "training_rows":len(train_set),"validation_rows":len(val_set),"excluded_rows":excluded_n,"scope":"distance <= 200 km and trip duration <= 6 hours","split":"random 80/20 seed 42",
         "example_prediction_min":example_output, "model_results":results},f,indent=2)
print(metrics.to_string(index=False));print("Example (5 km, hour 14):",example_output,"minutes")

# Chart 1: measured model comparison.
plt.figure(figsize=(9.6,5.2))
loc=np.arange(len(metrics))
plt.barh(loc,metrics.rmsle.values, color=["#BAC8D3","#386F9A","#116D72"])
plt.yticks(loc, ["Median baseline", "Distance only", "Distance + pickup hour"])
plt.xlabel("Validation RMSLE (lower is better)")
plt.xlim(0,max(metrics.rmsle)*1.2)
for i,v in enumerate(metrics.rmsle):plt.text(v+0.012,i,f"{v:.4f}",va="center")
plt.gca().invert_yaxis()
plt.title("NYC Taxi trip-duration model comparison")
plt.tight_layout();plt.savefig(REPORTS/"model_comparison.png",dpi=160);plt.close()

# Chart 2: hour-specific MAE; show a reliability dimension without retraining.
val=val_set[["pickup_hour","trip_duration"]].copy()
val["error_min"]=np.abs(yval_min-preds["Distance + pickup hour regression"])
hour=val.groupby("pickup_hour").agg(trips=("error_min","size"),mae_min=("error_min","mean")).reset_index()
hour.to_csv(REPORTS/"errors_by_pickup_hour.csv",index=False)
plt.figure(figsize=(9.6,5.2));plt.plot(hour.pickup_hour,hour.mae_min,marker="o",color="#116D72")
plt.xticks(range(24));plt.xlabel("Pickup hour (0–23)");plt.ylabel("Mean absolute error (minutes)")
plt.title("When is the model less reliable? Validation MAE by pickup hour")
plt.grid(alpha=.2);plt.tight_layout();plt.savefig(REPORTS/"errors_by_pickup_hour.png",dpi=160);plt.close()

# Chart 3: residual absolute error distribution, for business acceptability.
err=np.abs(yval_min-preds["Distance + pickup hour regression"])
q=np.percentile(err,[50,75,90,95]);
with open(REPORTS/"error_percentiles.json","w") as f:json.dump(dict(zip(["p50","p75","p90","p95"],[round(float(x),2) for x in q])),f,indent=2)
plt.figure(figsize=(9.6,5.2));plt.hist(np.minimum(err,30),bins=60,color="#386F9A")
plt.xlabel("Absolute prediction error (minutes; values >30 shown at 30)");plt.ylabel("Validation trips")
plt.title("Distribution of prediction errors (display capped at 30 minutes)")
plt.tight_layout();plt.savefig(REPORTS/"absolute_errors.png",dpi=160);plt.close()
print("Files saved in reports/ and artifacts/. Model reload confirmed.")
