"""
feature_engineering.py

Feature engineering module for the FinShield Fraud Detection Platform.

Responsibilities of this module:

1. Create additional behavioral and interaction features
2. Remove identifiers and leakage variables
3. Separate modeling dataset (X, y)
4. Preserve entity identifiers for traceability
5. Return categorical and numerical column lists for ML pipelines
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


# ------------------------------------------------
# Target
# ------------------------------------------------

TARGET_COLUMN = "is_fraud"


# ------------------------------------------------
# Identifier columns (kept for traceability only)
# ------------------------------------------------

ID_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
]


# ------------------------------------------------
# Columns that leak target information
# ------------------------------------------------

LEAKAGE_COLUMNS = [
    "fraud_risk_score",
    "fraud_probability",
]


# ------------------------------------------------
# Raw columns excluded from modeling
# ------------------------------------------------

RAW_EXCLUDE_COLUMNS = [
    "timestamp",
]


# ------------------------------------------------
# Categorical variables
# ------------------------------------------------

CATEGORICAL_COLUMNS = [
    "transaction_type",
    "merchant_category",
    "channel",
    "device_type",
    "transaction_city",
    "customer_home_city",
    "preferred_device_type",
]


# ------------------------------------------------
# Numerical variables
# ------------------------------------------------

NUMERICAL_COLUMNS = [
    "transaction_amount",
    "merchant_risk_score",
    "hour",
    "day_of_week",
    "is_weekend",
    "customer_tenure_days",
    "account_age_days",
    "avg_amount_30d",
    "std_amount_30d",
    "transactions_last_7d",
    "declines_last_30d",
    "chargebacks_last_90d",
    "amount_vs_avg_ratio",
    "is_high_amount",
    "is_very_high_amount",
    "is_night_transaction",
    "is_high_risk_merchant",
    "is_new_device",
    "is_foreign_transaction",
    "channel_risk",
    "category_risk",
    "high_amount_new_device",
    "foreign_high_risk_merchant",
    "web_night_transaction",
    "very_high_amount_foreign",

    # Additional engineered features
    "log_transaction_amount",
    "txn_velocity",
    "customer_risk_score",
    "decline_ratio",
    "amount_merchant_risk",
    "is_new_customer",
    "abnormal_activity",
]


# ------------------------------------------------
# Risk signal computation
# ------------------------------------------------

def compute_risk_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transaction-level risk signals from raw transaction and
    customer profile columns.

    This is the single source of truth for these signals: it is used
    both to build the synthetic fraud labels at data-generation time
    (see data_generation.assign_fraud_labels) and to featurize any new
    transaction at inference time. Keeping this logic in one place
    prevents train/serve skew between training data and live scoring.

    Expects the following columns to already be present in `df`:
    transaction_amount, avg_amount_30d, hour, merchant_risk_score,
    device_type, preferred_device_type, transaction_city,
    customer_home_city, channel, merchant_category.
    """

    data = df.copy()

    data["amount_vs_avg_ratio"] = data["transaction_amount"] / (data["avg_amount_30d"] + 1)
    data["is_high_amount"] = (data["amount_vs_avg_ratio"] > 2.8).astype(int)
    data["is_very_high_amount"] = (data["amount_vs_avg_ratio"] > 4.5).astype(int)
    data["is_night_transaction"] = data["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    data["is_high_risk_merchant"] = (data["merchant_risk_score"] > 0.75).astype(int)
    data["is_new_device"] = (data["device_type"] != data["preferred_device_type"]).astype(int)
    data["is_foreign_transaction"] = (data["transaction_city"] != data["customer_home_city"]).astype(int)

    data["channel_risk"] = data["channel"].map({
        "web": 1.20,
        "app": 0.75,
        "pos": 0.30,
    })

    data["category_risk"] = data["merchant_category"].map({
        "electronics": 1.10,
        "gaming": 1.00,
        "travel": 0.90,
        "fashion": 0.55,
        "restaurants": 0.25,
        "grocery": 0.20,
        "fuel": 0.15,
        "utilities": 0.10,
    })

    data["high_amount_new_device"] = (
        (data["is_high_amount"] == 1) & (data["is_new_device"] == 1)
    ).astype(int)

    data["foreign_high_risk_merchant"] = (
        (data["is_foreign_transaction"] == 1) & (data["is_high_risk_merchant"] == 1)
    ).astype(int)

    data["web_night_transaction"] = (
        (data["channel"] == "web") & (data["is_night_transaction"] == 1)
    ).astype(int)

    data["very_high_amount_foreign"] = (
        (data["is_very_high_amount"] == 1) & (data["is_foreign_transaction"] == 1)
    ).astype(int)

    return data


# ------------------------------------------------
# Feature creation
# ------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all behavioral, risk-signal, and interaction features
    used by the fraud detection model, starting from raw transaction
    and customer profile columns. Safe to call both on freshly
    generated training data and on a single new transaction at
    inference time.
    """

    data = compute_risk_signals(df)

    # ------------------------------------------------
    # Log-transformed transaction amount
    # ------------------------------------------------
    # Reduces skewness of heavy-tailed financial values
    data["log_transaction_amount"] = np.log1p(data["transaction_amount"])

    # ------------------------------------------------
    # Transaction velocity
    # ------------------------------------------------
    # Average number of transactions per day in the last week
    data["txn_velocity"] = data["transactions_last_7d"] / 7

    # ------------------------------------------------
    # Customer risk score
    # ------------------------------------------------
    # Combines declines and chargebacks
    data["customer_risk_score"] = (
        data["declines_last_30d"] * 0.5
        + data["chargebacks_last_90d"] * 1.5
    )

    # ------------------------------------------------
    # Decline ratio
    # ------------------------------------------------
    # Ratio of declines relative to recent activity
    # +1 prevents division by zero
    data["decline_ratio"] = (
        data["declines_last_30d"]
        / (data["transactions_last_7d"] + 1)
    )

    # ------------------------------------------------
    # Interaction: amount × merchant risk
    # ------------------------------------------------
    data["amount_merchant_risk"] = (
        data["transaction_amount"]
        * data["merchant_risk_score"]
    )

    # ------------------------------------------------
    # New customer flag
    # ------------------------------------------------
    data["is_new_customer"] = (
        data["customer_tenure_days"] < 90
    ).astype(int)

    # ------------------------------------------------
    # Abnormal activity
    # ------------------------------------------------
    data["abnormal_activity"] = (
        (data["transactions_last_7d"] > 15)
        & (data["amount_vs_avg_ratio"] > 2)
    ).astype(int)

    return data


# ------------------------------------------------
# Modeling dataset preparation
# ------------------------------------------------

def get_modeling_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[str]]:
    """
    Prepare datasets for ML modeling.

    Returns
    -------
    X : feature matrix
    y : target variable
    entity_df : dataframe with identifiers for traceability
    categorical_columns : list
    numerical_columns : list
    """

    data = create_features(df)

    # Preserve identifiers for traceability
    entity_cols = [col for col in ID_COLUMNS if col in data.columns]
    entity_df = data[entity_cols].copy()

    # Remove identifiers, leakage variables, and raw fields
    drop_cols = ID_COLUMNS + LEAKAGE_COLUMNS + RAW_EXCLUDE_COLUMNS
    existing_drop_cols = [col for col in drop_cols if col in data.columns]

    X = data.drop(columns=existing_drop_cols + [TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    # All categorical/numerical columns are expected to be present after
    # create_features(). Missing columns are a bug (e.g. malformed input
    # upstream) and should fail loudly rather than silently shrink the
    # feature set, which would otherwise cause silent train/serve skew.
    missing_categorical = [col for col in CATEGORICAL_COLUMNS if col not in X.columns]
    missing_numerical = [col for col in NUMERICAL_COLUMNS if col not in X.columns]
    if missing_categorical or missing_numerical:
        raise ValueError(
            "Modeling dataset is missing expected feature columns. "
            f"Missing categorical: {missing_categorical}. "
            f"Missing numerical: {missing_numerical}."
        )

    categorical_cols = list(CATEGORICAL_COLUMNS)
    numerical_cols = list(NUMERICAL_COLUMNS)

    return X, y, entity_df, categorical_cols, numerical_cols