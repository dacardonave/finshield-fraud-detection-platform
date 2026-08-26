"""
Tests for src/data_generation.py.

These use a small synthetic sample (not the full 100k-row dataset) so
the suite runs in a second or two, and check structural invariants
(shape, uniqueness, no missing values, reproducibility) rather than
exact statistical values, which would be too brittle for a random
sample this small.
"""

import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_customers, generate_dataset


@pytest.fixture(scope="module")
def small_dataset():
    config = SimulationConfig(n_customers=200, n_transactions=2000, random_state=42)
    return generate_dataset(config), config


def test_generate_customers_shape_and_uniqueness():
    config = SimulationConfig(n_customers=50, random_state=42)
    customers = generate_customers(config)

    assert len(customers) == 50
    assert customers["customer_id"].is_unique

    expected_columns = {
        "customer_id", "customer_home_city", "customer_tenure_days", "account_age_days",
        "preferred_device_type", "avg_amount_30d", "std_amount_30d",
        "transactions_last_7d", "declines_last_30d", "chargebacks_last_90d",
    }
    assert expected_columns.issubset(customers.columns)


def test_generate_dataset_shape_and_target(small_dataset):
    df, config = small_dataset
    assert len(df) == config.n_transactions
    assert "is_fraud" in df.columns
    assert set(df["is_fraud"].unique()).issubset({0, 1})


def test_fraud_rate_is_a_rare_minority_class(small_dataset):
    df, _ = small_dataset
    fraud_rate = df["is_fraud"].mean()
    # Not pinned to the ~3.3% observed on the full 100k dataset - a 2k-row
    # sample has more variance - just checked as a rare minority class,
    # consistent with the ~3% target in SimulationConfig.
    assert 0.0 < fraud_rate < 0.15


def test_no_missing_values_in_generated_dataset(small_dataset):
    df, _ = small_dataset
    assert df.isnull().sum().sum() == 0


def test_generation_is_deterministic_given_a_fixed_random_state():
    config = SimulationConfig(n_customers=50, n_transactions=200, random_state=42)
    df1 = generate_dataset(config)
    df2 = generate_dataset(config)
    pd.testing.assert_frame_equal(df1, df2)
