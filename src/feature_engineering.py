"""
feature_engineering.py

Feature engineering module for the FinShield Fraud Detection Platform.

This module prepares the dataset for machine learning by:

1. Creating additional behavioral and risk-based features
2. Removing identifiers and leakage variables
3. Separating features (X) and target (y)
4. Returning categorical and numerical feature lists for preprocessing pipelines
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
# Columns excluded from modeling
# ------------------------------------------------

ID_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
]

LEAKAGE_COLUMNS = [
    "fraud_risk_score",
    "fraud_probability",
]

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

    # new engineered features
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
    Create additional behavioral and interaction features.

    These features attempt to capture abnormal spending patterns,
    customer risk indicators, and merchant-related fraud signals.
    """

    data = df.copy()

    # --------------------------------------------
    # Log-transformed transaction amount
    # --------------------------------------------
    # Financial transaction values often follow a heavy-tailed distribution.
    # Applying log transformation stabilizes variance and reduces skewness.
    data["log_transaction_amount"] = np.log1p(data["transaction_amount"])

    # --------------------------------------------
    # Transaction velocity
    # --------------------------------------------
    # Average number of transactions per day in the last week.
    # Sudden spikes in velocity may indicate automated or fraudulent activity.
    data["txn_velocity"] = data["transactions_last_7d"] / 7

    # --------------------------------------------
    # Customer risk score
    # --------------------------------------------
    # Composite indicator combining declines and chargebacks.
    # Prior payment issues often correlate with future fraud attempts.
    data["customer_risk_score"] = (
        data["declines_last_30d"] * 0.5
        + data["chargebacks_last_90d"] * 1.5
    )

    # --------------------------------------------
    # Decline ratio
    # --------------------------------------------
    # Ratio of declined transactions to recent transaction activity.
    # +1 is added to the denominator to avoid division by zero
    # and stabilize the metric for low activity customers.
    data["decline_ratio"] = (
        data["declines_last_30d"]
        / (data["transactions_last_7d"] + 1)
    )

    # --------------------------------------------
    # Amount × merchant risk interaction
    # --------------------------------------------
    # High-value transactions at high-risk merchants
    # are more suspicious than either factor alone.
    data["amount_merchant_risk"] = (
        data["transaction_amount"]
        * data["merchant_risk_score"]
    )

    # --------------------------------------------
    # New customer indicator
    # --------------------------------------------
    # Fraud often targets recently created accounts.
    data["is_new_customer"] = (
        data["customer_tenure_days"] < 90
    ).astype(int)

    # --------------------------------------------
    # Abnormal activity flag
    # --------------------------------------------
    # Captures unusually high recent activity combined
    # with spending significantly above historical average.
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
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare X, y, categorical columns, and numerical columns
    for machine learning models.
    """

    data = create_features(df)

    # Remove identifiers and leakage variables
    drop_cols = ID_COLUMNS + LEAKAGE_COLUMNS + RAW_EXCLUDE_COLUMNS
    existing_drop_cols = [col for col in drop_cols if col in data.columns]

    X = data.drop(columns=existing_drop_cols + [TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    # Identify categorical and numerical columns present in X
    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    num_cols = [col for col in NUMERICAL_COLUMNS if col in X.columns]

    return X, y, cat_cols, num_cols