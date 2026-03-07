"""
data_generation.py

Synthetic data generation module for the FinShield Fraud Detection Platform.

This module is responsible for:
1. Creating synthetic customer profiles
2. Simulating financial transactions with business-aware patterns
3. Generating fraud labels based on synthetic business rules
4. Saving the generated dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    output_path: str = "data/raw/transactions.csv"


# -----------------------------
# Customer generation
# -----------------------------

def generate_customers(config: SimulationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)

    customer_ids = [f"CUST_{i:06d}" for i in range(1, config.n_customers + 1)]

    cities = ["New York", "Newark", "Jersey City", "Brooklyn", "Queens", "Bronx"]
    device_types = ["ios", "android", "desktop"]

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_home_city": rng.choice(cities, size=config.n_customers),
        "customer_tenure_days": rng.integers(30, 3650, size=config.n_customers),
        "account_age_days": rng.integers(30, 3650, size=config.n_customers),
        "preferred_device_type": rng.choice(device_types, size=config.n_customers, p=[0.45, 0.40, 0.15]),
        "avg_amount_30d": rng.uniform(15, 250, size=config.n_customers).round(2),
        "std_amount_30d": rng.uniform(5, 80, size=config.n_customers).round(2),
        "transactions_last_7d": rng.integers(1, 25, size=config.n_customers),
        "declines_last_30d": rng.integers(0, 5, size=config.n_customers),
        "chargebacks_last_90d": rng.integers(0, 3, size=config.n_customers),
    })

    return customers


# -----------------------------
# Helpers
# -----------------------------

def generate_amounts_by_category(categories: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """
    Generate transaction amounts based on merchant category.
    """
    category_params = {
        "grocery": {"mean": 3.5, "sigma": 0.45},
        "electronics": {"mean": 4.4, "sigma": 0.65},
        "travel": {"mean": 4.6, "sigma": 0.70},
        "fashion": {"mean": 4.0, "sigma": 0.60},
        "gaming": {"mean": 3.2, "sigma": 0.55},
        "restaurants": {"mean": 3.4, "sigma": 0.50},
        "fuel": {"mean": 3.3, "sigma": 0.35},
        "utilities": {"mean": 3.7, "sigma": 0.30},
    }

    amounts = []
    for category in categories:
        params = category_params[category]
        amount = rng.lognormal(mean=params["mean"], sigma=params["sigma"])
        amounts.append(round(amount, 2))

    return np.array(amounts)


def assign_merchant_risk_by_category(categories: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """
    Generate merchant risk scores influenced by merchant category.
    """
    category_risk_ranges = {
        "grocery": (0.05, 0.35),
        "electronics": (0.30, 0.95),
        "travel": (0.25, 0.85),
        "fashion": (0.20, 0.75),
        "gaming": (0.35, 0.90),
        "restaurants": (0.10, 0.50),
        "fuel": (0.08, 0.40),
        "utilities": (0.03, 0.20),
    }

    scores = []
    for category in categories:
        low, high = category_risk_ranges[category]
        scores.append(round(rng.uniform(low, high), 3))

    return np.array(scores)


# -----------------------------
# Transaction generation
# -----------------------------

def generate_transactions(customers: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)

    merchant_categories = [
        "grocery", "electronics", "travel", "fashion",
        "gaming", "restaurants", "fuel", "utilities"
    ]
    transaction_types = ["card", "digital_payment"]
    channels = ["app", "web", "pos"]
    channel_probs = [0.50, 0.30, 0.20]
    cities = ["New York", "Newark", "Jersey City", "Brooklyn", "Queens", "Miami", "Los Angeles"]
    device_types = ["ios", "android", "desktop"]

    sampled_customers = customers.sample(
        n=config.n_transactions,
        replace=True,
        random_state=config.random_state
    ).reset_index(drop=True)

    base_dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="h")
    sampled_timestamps = rng.choice(base_dates, size=config.n_transactions)

    sampled_categories = rng.choice(
        merchant_categories,
        size=config.n_transactions,
        p=[0.20, 0.10, 0.08, 0.12, 0.10, 0.18, 0.12, 0.10]
    )

    transaction_amounts = generate_amounts_by_category(pd.Series(sampled_categories), rng)
    merchant_risk_scores = assign_merchant_risk_by_category(pd.Series(sampled_categories), rng)

    transactions = pd.DataFrame({
        "transaction_id": [f"TXN_{i:07d}" for i in range(1, config.n_transactions + 1)],
        "customer_id": sampled_customers["customer_id"].values,
        "timestamp": sampled_timestamps,
        "transaction_type": rng.choice(transaction_types, size=config.n_transactions, p=[0.65, 0.35]),
        "merchant_category": sampled_categories,
        "transaction_amount": transaction_amounts,
        "merchant_id": [f"MERCH_{i:05d}" for i in rng.integers(1, 500, size=config.n_transactions)],
        "channel": rng.choice(channels, size=config.n_transactions, p=channel_probs),
        "device_type": rng.choice(device_types, size=config.n_transactions, p=[0.45, 0.40, 0.15]),
        "device_id": [f"DEV_{i:06d}" for i in rng.integers(1, 20000, size=config.n_transactions)],
        "transaction_city": rng.choice(cities, size=config.n_transactions),
        "merchant_risk_score": merchant_risk_scores,
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
    df = transactions.copy()

    # Core rule-based features
    df["amount_vs_avg_ratio"] = df["transaction_amount"] / (df["avg_amount_30d"] + 1)
    df["is_high_amount"] = (df["amount_vs_avg_ratio"] > 2.8).astype(int)
    df["is_very_high_amount"] = (df["amount_vs_avg_ratio"] > 4.5).astype(int)
    df["is_night_transaction"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    df["is_high_risk_merchant"] = (df["merchant_risk_score"] > 0.75).astype(int)
    df["is_new_device"] = (df["device_type"] != df["preferred_device_type"]).astype(int)
    df["is_foreign_transaction"] = (df["transaction_city"] != df["customer_home_city"]).astype(int)

    # Base channel risk
    df["channel_risk"] = df["channel"].map({
        "web": 1.20,
        "app": 0.75,
        "pos": 0.30
    })

    # Base merchant category risk
    df["category_risk"] = df["merchant_category"].map({
        "electronics": 1.10,
        "gaming": 1.00,
        "travel": 0.90,
        "fashion": 0.55,
        "restaurants": 0.25,
        "grocery": 0.20,
        "fuel": 0.15,
        "utilities": 0.10
    })

    # Interaction terms
    df["high_amount_new_device"] = (
        (df["is_high_amount"] == 1) & (df["is_new_device"] == 1)
    ).astype(int)

    df["foreign_high_risk_merchant"] = (
        (df["is_foreign_transaction"] == 1) & (df["is_high_risk_merchant"] == 1)
    ).astype(int)

    df["web_night_transaction"] = (
        (df["channel"] == "web") & (df["is_night_transaction"] == 1)
    ).astype(int)

    df["very_high_amount_foreign"] = (
        (df["is_very_high_amount"] == 1) & (df["is_foreign_transaction"] == 1)
    ).astype(int)

    # Continuous latent risk score
    df["fraud_risk_score"] = (
        -4.2
        + 0.90 * df["is_high_amount"]
        + 1.20 * df["is_very_high_amount"]
        + 0.45 * df["is_night_transaction"]
        + 0.70 * df["is_high_risk_merchant"]
        + 0.55 * df["is_new_device"]
        + 0.60 * df["is_foreign_transaction"]
        + df["channel_risk"]
        + df["category_risk"]
        + 1.00 * df["high_amount_new_device"]
        + 1.10 * df["foreign_high_risk_merchant"]
        + 0.75 * df["web_night_transaction"]
        + 1.25 * df["very_high_amount_foreign"]
    )

    rng = np.random.default_rng(config.random_state)

    # Add noise for stochasticity
    noise = rng.normal(0, 0.35, size=len(df))
    df["fraud_risk_score"] = df["fraud_risk_score"] + noise

    # Convert score to probability using sigmoid
    df["fraud_probability"] = 1 / (1 + np.exp(-df["fraud_risk_score"]))

    # Optional scaling to keep overall fraud rate in a reasonable band
    df["fraud_probability"] = df["fraud_probability"] * 0.18

    # Clip probabilities
    df["fraud_probability"] = df["fraud_probability"].clip(0.0005, 0.95)

    # Sample final label from probability
    df["is_fraud"] = rng.binomial(1, df["fraud_probability"])

    return df


# -----------------------------
# Save dataset
# -----------------------------

def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


# -----------------------------
# Pipeline runner
# -----------------------------

def generate_dataset(config: SimulationConfig) -> pd.DataFrame:
    customers = generate_customers(config)
    transactions = generate_transactions(customers, config)
    dataset = assign_fraud_labels(transactions, config)
    return dataset


def print_summary(df: pd.DataFrame) -> None:
    print("\nDataset generated successfully.")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    print("\nFraud distribution:")
    print(df["is_fraud"].value_counts(normalize=True).rename("proportion"))
    print("\nChannel distribution:")
    print(df["channel"].value_counts(normalize=True).round(4))
    print("\nTransaction type distribution:")
    print(df["transaction_type"].value_counts(normalize=True).round(4))
    print("\nFraud by channel:")
    print(df.groupby("channel")["is_fraud"].mean().round(4).sort_values(ascending=False))
    print("\nFraud by merchant category:")
    print(df.groupby("merchant_category")["is_fraud"].mean().round(4).sort_values(ascending=False))


# -----------------------------
# Main execution
# -----------------------------

if __name__ == "__main__":
    config = SimulationConfig()
    df = generate_dataset(config)
    save_dataset(df, config.output_path)
    print_summary(df)
    print(f"\nFile saved to: {config.output_path}")