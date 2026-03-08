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
# Feature creation
# ------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional behavioral and interaction features
    used by the fraud detection model.
    """

    data = df.copy()

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

    # Identify categorical and numerical columns present in X
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    numerical_cols = [col for col in NUMERICAL_COLUMNS if col in X.columns]

    return X, y, entity_df, categorical_cols, numerical_cols