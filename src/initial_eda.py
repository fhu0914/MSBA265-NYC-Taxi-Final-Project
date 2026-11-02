from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"  

print("Project root:", ROOT)
print("Data folder:", DATA_DIR)

TRAIN_ZIP = DATA_DIR / "train.zip"

train = pd.read_csv(TRAIN_ZIP)

print("\nTrain shape:", train.shape)
print("\nColumns:")
print(train.columns.tolist())

print("\nData types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum())

print("\nFirst 5 rows:")
print(train.head())

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["dropoff_datetime"] = pd.to_datetime(train["dropoff_datetime"])

print("\nDatetime types after conversion:")
print(train[["pickup_datetime", "dropoff_datetime"]].dtypes)    

train["pickup_hour"] = train["pickup_datetime"].dt.hour
hourly_median = train.groupby("pickup_hour")["trip_duration"].median()

lat1 = np.radians(train["pickup_latitude"])
lon1 = np.radians(train["pickup_longitude"])
lat2 = np.radians(train["dropoff_latitude"])
lon2 = np.radians(train["dropoff_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
)

train["distance_km"] = 2 * 6371.0 * np.arcsin(np.sqrt(a))

print("\nTrip distance summary (km):")
print(train["distance_km"].describe())

print("\nTrip duration summary (seconds):")
print(train["trip_duration"].describe())

print("\nTrip duration median:")
print(train["trip_duration"].median())

plt.figure(figsize=(8, 5))
plt.hist(train["trip_duration"], bins=100)
plt.xlabel("Trip Duration (seconds)")
plt.ylabel("Number of Trips")
plt.title("Distribution of NYC Taxi Trip Duration")
plt.tight_layout()
plt.show()  

plt.figure(figsize=(8, 5))
plt.hist(np.log1p(train["trip_duration"]), bins=100)
plt.xlabel("log(1 + Trip Duration)")
plt.ylabel("Number of Trips")
plt.title("Distribution of Log-Transformed NYC Taxi Trip Duration")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "trip_duration_log_distribution.png",
    dpi=150
)

plt.show()

plt.figure(figsize=(8, 5))
hourly_median.plot(kind="bar")

plt.xlabel("Pickup Hour")
plt.ylabel("Median Trip Duration (seconds)")
plt.title("Median Trip Duration by Pickup Hour")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "median_trip_duration_by_hour.png",
    dpi=150
)

plt.show()

train_part, val_part = train_test_split(
    train,
    test_size=0.20,
    random_state=42
)

print("\nSplit sizes:")
print("Training:", len(train_part))
print("Validation:", len(val_part))

baseline_prediction = train_part["trip_duration"].median()

print("\nBaseline prediction (median trip duration):")
print(baseline_prediction)

baseline_preds = np.full(len(val_part), baseline_prediction)

baseline_rmsle = np.sqrt(
    np.mean(
        (
            np.log1p(baseline_preds)
            - np.log1p(val_part["trip_duration"])
        ) ** 2
    )
)

print("\nBaseline RMSLE on validation:")
print(baseline_rmsle)

print("\nTrip distance summary (km):")
print(train["distance_km"].describe())

X_train = train_part[["distance_km"]]
X_val = val_part[["distance_km"]]

y_train_log = np.log1p(train_part["trip_duration"])

distance_model = LinearRegression()
distance_model.fit(X_train, y_train_log)

val_pred_log = distance_model.predict(X_val)

distance_rmsle = np.sqrt(
    np.mean(
        (
            val_pred_log
            - np.log1p(val_part["trip_duration"])
        ) ** 2
    )
)

print("\nDistance-only linear regression RMSLE:")
print(distance_rmsle)

X_train_2 = train_part[["distance_km", "pickup_hour"]]
X_val_2 = val_part[["distance_km", "pickup_hour"]]

distance_hour_model = LinearRegression()
distance_hour_model.fit(X_train_2, y_train_log)

val_pred_log_2 = distance_hour_model.predict(X_val_2)

distance_hour_rmsle = np.sqrt(
    np.mean(
        (
            val_pred_log_2
            - np.log1p(val_part["trip_duration"])
        ) ** 2
    )
)

print("\nDistance + pickup hour linear regression RMSLE:")
print(distance_hour_rmsle)