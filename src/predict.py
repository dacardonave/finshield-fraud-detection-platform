"""
predict.py

Inference module for the FinShield Fraud Detection Platform.

Loads the trained pipeline once and scores new transactions through the
exact same feature pipeline used at training time
(feature_engineering.create_features, built on top of
feature_engineering.compute_risk_signals). This is what guarantees
train/serve parity: a live transaction is featurized identically to a
training row.

This module is the single place that should ever call the model's
predict_proba. The FastAPI service (api/main.py) and any future batch
scoring job must call into predict_one / predict_batch rather than
reimplementing scoring logic - that duplication is exactly what caused the
train/serve skew bug fixed in feature_engineering.py.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from src.feature_engineering import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, create_features
from src.utils import MODEL_DIR, get_logger, load_model_metadata

logger = get_logger(__name__)

MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

# Raw fields a caller must supply for a single transaction. Everything else
# the model needs (risk signals, behavioral features) is derived internally
# by create_features - see feature_engineering.compute_risk_signals. This
# list is also the basis for the API's request schema (api/main.py).
REQUIRED_RAW_FIELDS = [
    "transaction_type",
    "merchant_category",
    "transaction_amount",
    "channel",
    "device_type",
    "transaction_city",
    "merchant_risk_score",
    "timestamp",
    "customer_home_city",
    "customer_tenure_days",
    "account_age_days",
    "preferred_device_type",
    "avg_amount_30d",
    "std_amount_30d",
    "transactions_last_7d",
    "declines_last_30d",
    "chargebacks_last_90d",
]


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained pipeline. Raises a clear error if training hasn't run yet."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    logger.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    """Load the business-cost-optimal decision threshold selected during training."""
    metadata = load_model_metadata()
    return float(metadata["decision_threshold"])


def risk_tier(probability: float, threshold: float) -> str:
    """
    Bucket a fraud probability into a human-readable tier, scaled relative
    to the decision threshold rather than fixed cutoffs, so the tiers stay
    meaningful if the model or threshold is retrained.
    """
    if probability < threshold * 0.5:
        return "low"
    if probability < threshold:
        return "medium"
    if probability < threshold * 1.5:
        return "high"
    return "critical"


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    missing_raw = [c for c in REQUIRED_RAW_FIELDS if c not in data.columns]
    if missing_raw:
        raise ValueError(f"Missing required raw transaction fields: {missing_raw}")

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    data = create_features(data)

    missing_features = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing_features:
        # Should be unreachable given the checks above and in
        # feature_engineering.py, but fail loudly rather than silently
        # score on a partial feature set if it ever happens.
        raise ValueError(f"Feature computation did not produce expected columns: {missing_features}")

    return data[FEATURE_COLUMNS]


PREDICTION_OUTPUT_COLUMNS = ["fraud_probability", "is_fraud", "risk_tier", "decision_threshold"]


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch of transactions.

    Returns the input DataFrame with fraud_probability, is_fraud, risk_tier,
    and decision_threshold columns appended. If the input already has a
    column with one of those names (e.g. a ground-truth `is_fraud` column
    passed in for backtesting), this raises rather than silently
    overwriting it - drop or rename that column before calling.
    """
    colliding = [c for c in PREDICTION_OUTPUT_COLUMNS if c in df.columns]
    if colliding:
        raise ValueError(
            f"Input DataFrame already has column(s) {colliding}, which "
            "predict_batch would overwrite with its own output. Rename or "
            "drop them first (e.g. keep a ground-truth `is_fraud` column as "
            "`y_true` for backtesting)."
        )

    model = load_model()
    threshold = load_threshold()

    X = _prepare_features(df)
    proba = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["fraud_probability"] = proba
    result["is_fraud"] = proba >= threshold
    result["risk_tier"] = [risk_tier(p, threshold) for p in proba]
    result["decision_threshold"] = threshold
    return result


def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single transaction, given as a dict of raw fields
    (see REQUIRED_RAW_FIELDS). Returns the input fields plus
    fraud_probability, is_fraud, risk_tier, and decision_threshold.
    """
    df = pd.DataFrame([payload])
    result = predict_batch(df).iloc[0].to_dict()
    result["is_fraud"] = bool(result["is_fraud"])
    result["fraud_probability"] = float(result["fraud_probability"])
    result["decision_threshold"] = float(result["decision_threshold"])
    return result
