"""
conftest.py

Shared pytest fixtures for the FinShield test suite: two example
transactions (one clearly legitimate, one clearly suspicious) used by
both the predict.py tests and the API tests, so both layers are tested
against the same known scenarios instead of subtly different ones.
"""

import pytest

LEGIT_TRANSACTION = {
    "transaction_type": "card",
    "merchant_category": "grocery",
    "transaction_amount": 45.0,
    "channel": "pos",
    "device_type": "ios",
    "transaction_city": "Brooklyn",
    "merchant_risk_score": 0.15,
    "timestamp": "2025-06-15T14:00:00",
    "customer_home_city": "Brooklyn",
    "customer_tenure_days": 900,
    "account_age_days": 900,
    "preferred_device_type": "ios",
    "avg_amount_30d": 50.0,
    "std_amount_30d": 20.0,
    "transactions_last_7d": 5,
    "declines_last_30d": 0,
    "chargebacks_last_90d": 0,
}

SUSPICIOUS_TRANSACTION = {
    **LEGIT_TRANSACTION,
    "merchant_category": "electronics",
    "transaction_amount": 900.0,
    "channel": "web",
    "device_type": "android",
    "transaction_city": "Miami",
    "merchant_risk_score": 0.9,
    "timestamp": "2025-06-15T02:30:00",
}


@pytest.fixture
def legit_transaction() -> dict:
    return dict(LEGIT_TRANSACTION)


@pytest.fixture
def suspicious_transaction() -> dict:
    return dict(SUSPICIOUS_TRANSACTION)
