"""
Tests for api/main.py, using FastAPI's TestClient (no real server needed).

Covers the two-layer validation design documented in api/main.py: Pydantic
request validation (missing field, invalid category, out-of-range value)
and the happy path end to end through the real trained model.
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] is not None
    assert body["decision_threshold"] is not None


def test_predict_endpoint_with_a_valid_transaction(legit_transaction):
    response = client.post("/predict", json=legit_transaction)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["risk_tier"] in {"low", "medium", "high", "critical"}


def test_predict_endpoint_ranks_suspicious_above_legit(legit_transaction, suspicious_transaction):
    legit_response = client.post("/predict", json=legit_transaction).json()
    suspicious_response = client.post("/predict", json=suspicious_transaction).json()
    assert suspicious_response["fraud_probability"] > legit_response["fraud_probability"]


def test_predict_endpoint_missing_field_returns_422():
    response = client.post("/predict", json={"transaction_type": "card"})
    assert response.status_code == 422


def test_predict_endpoint_invalid_category_returns_422(legit_transaction):
    payload = dict(legit_transaction)
    payload["merchant_category"] = "not_a_real_category"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_negative_amount_returns_422(legit_transaction):
    payload = dict(legit_transaction)
    payload["transaction_amount"] = -5.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
