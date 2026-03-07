"""
Synthetic data generation module for the FinShield Fraud Detection Platform.

This module is responsible for:
1. Creating synthetic customer profiles
2. Simulating financial transactions
3. Generating fraud labels based on business rules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class SimulationConfig:
    n_customers: int = 5000
    n_transactions: int = 100000
    fraud_rate_target: float = 0.03
    random_state: int = 42


# -----------------------------
# Customer generation
# -----------------------------

def generate_customers(config: SimulationConfig) -> pd.DataFrame:
    """
    Generate synthetic customer profiles.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic customer-level information.
    """
    rng = np.random.default_rng(config.random_state)

    customer_ids = [f"CUST_{i:06d}" for i in range(1, config.n_customers + 1)]

    cities = ["New York", "Newark", "Jersey City", "Brooklyn", "Queens", "Bronx"]
    device_types = ["ios", "android", "desktop"]

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_home_city": rng.choice(cities, size=config.n_customers),
        "customer_tenure_days": rng.integers(30, 3650, size=config.n_customers),
        "account_age_days": rng.integers(30, 3650, size=config.n_customers),
        "preferred_device_type": rng.choice(device_types, size=config.n_customers, p=[0.4, 0.4, 0.2]),
        "avg_amount_30d": rng.uniform(10, 500, size=config.n_customers).round(2),
        "std_amount_30d": rng.uniform(5, 150, size=config.n_customers).round(2),
        "transactions_last_7d": rng.integers(1, 25, size=config.n_customers),
        "declines_last_30d": rng.integers(0, 5, size=config.n_customers),
        "chargebacks_last_90d": rng.integers(0, 3, size=config.n_customers),
    })

    return customers


# -----------------------------
# Transaction generation
# -----------------------------

def generate_transactions(customers: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """
    Generate synthetic transaction-level data.

    Parameters
    ----------
    customers : pd.DataFrame
        Customer profiles.
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic transaction-level data.
    """
    rng = np.random.default_rng(config.random_state)

    merchant_categories = [
        "grocery", "electronics", "travel", "fashion",
        "gaming", "restaurants", "fuel", "utilities"
    ]
    channels = ["app", "web", "pos"]
    transaction_types = ["card", "digital_payment"]
    cities = ["New York", "Newark", "Jersey City", "Brooklyn", "Queens", "Miami", "Los Angeles"]
    device_types = ["ios", "android", "desktop"]

    sampled_customers = customers.sample(
        n=config.n_transactions,
        replace=True,
        random_state=config.random_state
    ).reset_index(drop=True)

    base_dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="H")
    sampled_timestamps = rng.choice(base_dates, size=config.n_transactions)

    transactions = pd.DataFrame({
        "transaction_id": [f"TXN_{i:07d}" for i in range(1, config.n_transactions + 1)],
        "customer_id": sampled_customers["customer_id"].values,
        "timestamp": sampled_timestamps,
        "transaction_type": rng.choice(transaction_types, size=config.n_transactions, p=[0.65, 0.35]),
        "transaction_amount": rng.lognormal(mean=3.7, sigma=0.8, size=config.n_transactions).round(2),
        "merchant_category": rng.choice(merchant_categories, size=config.n_transactions),
        "merchant_id": [f"MERCH_{i:05d}" for i in rng.integers(1, 500, size=config.n_transactions)],
        "channel": rng.choice(channels, size=config.n_transactions, p=[0.5, 0.25, 0.25]),
        "device_type": rng.choice(device_types, size=config.n_transactions, p=[0.4, 0.4, 0.2]),
        "device_id": [f"DEV_{i:06d}" for i in rng.integers(1, 20000, size=config.n_transactions)],
        "transaction_city": rng.choice(cities, size=config.n_transactions),
        "merchant_risk_score": rng.uniform(0.01, 0.99, size=config.n_transactions).round(3),
    })

    transactions["hour"] = pd.to_datetime(transactions["timestamp"]).dt.hour
    transactions["day_of_week"] = pd.to_datetime(transactions["timestamp"]).dt.dayofweek
    transactions["is_weekend"] = transactions["day_of_week"].isin([5, 6]).astype(int)

    transactions = transactions.merge(customers, on="customer_id", how="left")

    return transactions


# -----------------------------
# Fraud labeling
# -----------------------------

def assign_fraud_labels(transactions: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """
    Assign fraud labels using synthetic business rules.

    Parameters
    ----------
    transactions : pd.DataFrame
        Transaction-level dataset.
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    pd.DataFrame
        Transactions with fraud labels and intermediate risk score.
    """
    df = transactions.copy()

    df["amount_vs_avg_ratio"] = df["transaction_amount"] / (df["avg_amount_30d"] + 1)
    df["is_high_amount"] = (df["amount_vs_avg_ratio"] > 3.0).astype(int)
    df["is_night_transaction"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    df["is_high_risk_merchant"] = (df["merchant_risk_score"] > 0.8).astype(int)
    df["is_new_device"] = (df["device_type"] != df["preferred_device_type"]).astype(int)
    df["is_foreign_transaction"] = (df["transaction_city"] != df["customer_home_city"]).astype(int)

    # Synthetic risk score
    df["fraud_risk_score"] = (
        0.30 * df["is_high_amount"]
        + 0.15 * df["is_night_transaction"]
        + 0.20 * df["is_high_risk_merchant"]
        + 0.15 * df["is_new_device"]
        + 0.20 * df["is_foreign_transaction"]
    )

    rng = np.random.default_rng(config.random_state)
    noise = rng.uniform(0, 0.25, size=len(df))
    df["fraud_risk_score"] = df["fraud_risk_score"] + noise

    threshold = np.quantile(df["fraud_risk_score"], 1 - config.fraud_rate_target)
    df["is_fraud"] = (df["fraud_risk_score"] >= threshold).astype(int)

    return df


# -----------------------------
# Pipeline runner
# -----------------------------

def generate_dataset(config: SimulationConfig) -> pd.DataFrame:
    """
    Run the full synthetic data generation pipeline.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    pd.DataFrame
        Final transaction dataset with fraud labels.
    """
    customers = generate_customers(config)
    transactions = generate_transactions(customers, config)
    dataset = assign_fraud_labels(transactions, config)

    return dataset


# -----------------------------
# Main execution
# -----------------------------

if __name__ == "__main__":
    config = SimulationConfig()
    df = generate_dataset(config)

    print("Dataset generated successfully.")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nFraud rate:")
    print(df["is_fraud"].mean().round(4))