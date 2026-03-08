"""
feature_engineering.py

Feature engineering module for the FinShield Fraud Detection Platform.
"""

from __future__ import annotations

from typing import List, Tuple
import pandas as pd


TARGET_COLUMN = "is_fraud"

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

CATEGORICAL_COLUMNS = [
    "transaction_type",
    "merchant_category",
    "channel",
    "device_type",
    "transaction_city",
    "customer_home_city",
    "preferred_device_type",
]

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
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataset ready for modeling.
    Assumes the simulation pipeline already created the engineered columns.
    """
    data = df.copy()

    required_columns = set(CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [TARGET_COLUMN])
    missing_cols = required_columns - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return data


def get_modeling_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare X, y, categorical columns, and numerical columns for modeling.
    """
    data = create_features(df)

    drop_cols = ID_COLUMNS + LEAKAGE_COLUMNS + RAW_EXCLUDE_COLUMNS
    existing_drop_cols = [col for col in drop_cols if col in data.columns]

    X = data.drop(columns=existing_drop_cols + [TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    num_cols = [col for col in NUMERICAL_COLUMNS if col in X.columns]

    return X, y, cat_cols, num_cols