"""
Tests for src/predict.py.

These run against the actual trained model committed at
models/model.joblib (see Phase 2) rather than a mock, so they double as
a regression check on the real production artifact - if retraining ever
changes the model in a way that breaks the prediction contract, these
tests catch it.
"""

import pandas as pd
import pytest

from src.predict import REQUIRED_RAW_FIELDS, predict_batch, predict_one, risk_tier


def test_predict_one_returns_the_expected_shape(legit_transaction):
    result = predict_one(legit_transaction)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)
    assert result["risk_tier"] in {"low", "medium", "high", "critical"}
    assert result["decision_threshold"] > 0


def test_predict_one_ranks_suspicious_above_legit(legit_transaction, suspicious_transaction):
    legit = predict_one(legit_transaction)
    suspicious = predict_one(suspicious_transaction)
    assert suspicious["fraud_probability"] > legit["fraud_probability"]


def test_predict_one_raises_on_missing_field(legit_transaction):
    broken = dict(legit_transaction)
    del broken["merchant_risk_score"]
    with pytest.raises(ValueError):
        predict_one(broken)


def test_predict_batch_rejects_a_colliding_is_fraud_column(legit_transaction):
    df = pd.DataFrame([legit_transaction])
    df["is_fraud"] = 0  # would collide with the model's own output column
    with pytest.raises(ValueError):
        predict_batch(df)


def test_predict_batch_scores_multiple_rows(legit_transaction, suspicious_transaction):
    df = pd.DataFrame([legit_transaction, suspicious_transaction])
    result = predict_batch(df)
    assert len(result) == 2
    assert result["fraud_probability"].between(0, 1).all()


def test_risk_tier_buckets_scale_with_threshold():
    threshold = 0.5
    assert risk_tier(0.1, threshold) == "low"
    assert risk_tier(0.4, threshold) == "medium"
    assert risk_tier(0.6, threshold) == "high"
    assert risk_tier(0.9, threshold) == "critical"


def test_required_raw_fields_match_the_example_fixture(legit_transaction):
    assert set(REQUIRED_RAW_FIELDS) == set(legit_transaction.keys())
