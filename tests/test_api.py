"""Run: python -m pytest -q tests/test_api.py"""
from fastapi.testclient import TestClient
from app import app
client=TestClient(app)

def test_health():
    r=client.get("/health")
    assert r.status_code==200 and r.json()["model_available"] is True

def test_valid_prediction():
    r=client.post("/predict",json={"distance_km":5,"pickup_hour":14})
    assert r.status_code==200
    assert r.json()["predicted_minutes"]>0

def test_invalid_hour():
    r=client.post("/predict",json={"distance_km":5,"pickup_hour":31})
    assert r.status_code==422

def test_invalid_distance():
    r=client.post("/predict",json={"distance_km":-3,"pickup_hour":14})
    assert r.status_code==422
