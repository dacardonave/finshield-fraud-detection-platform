"""
Tests for src/feature_engineering.py.

These specifically exercise the Phase 1 refactor: compute_risk_signals as
the single shared source of the risk-signal features, and get_modeling_data
failing loudly instead of silently dropping columns - the exact bug this
project's audit found and fixed.
"""

import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_dataset
from src.feature_engineering import (
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERICAL_COLUMNS,
    compute_risk_signals,
    create_features,
    get_modeling_data,
)


@pytest.fixture(scope="module")
def small_df():
    config = SimulationConfig(n_customers=100, n_transactions=500, random_state=42)
    return generate_dataset(config)


def test_compute_risk_signals_adds_expected_columns(small_df):
    result = compute_risk_signals(small_df)
    expected = {
        "amount_vs_avg_ratio", "is_high_amount", "is_very_high_amount",
        "is_night_transaction", "is_high_risk_merchant", "is_new_device",
        "is_foreign_transaction", "channel_risk", "category_risk",
        "high_amount_new_device", "foreign_high_risk_merchant",
        "web_night_transaction", "very_high_amount_foreign",
    }
    assert expected.issubset(result.columns)


def test_compute_risk_signals_requires_raw_columns(small_df):
    broken = small_df.drop(columns=["merchant_risk_score"])
    with pytest.raises(KeyError):
        compute_risk_signals(broken)


def test_create_features_matches_data_generation_exactly(small_df):
    """
    data_generation.assign_fraud_labels already computed the risk signals
    once (via compute_risk_signals) when it built `small_df`.
    create_features recomputes them independently here. The values must
    match exactly - if they diverged, training-time features would no
    longer match what the synthetic labels were actually generated from.
    """
    recomputed = create_features(small_df)
    for col in ["amount_vs_avg_ratio", "channel_risk", "category_risk", "is_high_amount"]:
        pd.testing.assert_series_equal(
            recomputed[col].reset_index(drop=True),
            small_df[col].reset_index(drop=True),
            check_names=False,
        )


def test_get_modeling_data_drops_ids_and_leakage_columns(small_df):
    X, y, entity_df, categorical_cols, numerical_cols = get_modeling_data(small_df)

    for col in ID_COLUMNS + LEAKAGE_COLUMNS:
        assert col not in X.columns
    assert set(entity_df.columns) == set(ID_COLUMNS)
    assert set(y.unique()).issubset({0, 1})


def test_get_modeling_data_returns_the_full_expected_feature_set(small_df):
    X, y, entity_df, categorical_cols, numerical_cols = get_modeling_data(small_df)

    assert set(categorical_cols) == set(CATEGORICAL_COLUMNS)
    assert set(numerical_cols) == set(NUMERICAL_COLUMNS)
    assert set(categorical_cols + numerical_cols) == set(X.columns)


def test_get_modeling_data_fails_loudly_on_a_missing_column(small_df):
    """
    This is the exact bug fixed in Phase 1: before the fix, a missing
    upstream column was silently dropped from the feature set instead of
    raising. Now it must fail - either compute_risk_signals raises first
    (KeyError) or get_modeling_data's own explicit check does (ValueError).
    """
    broken = small_df.drop(columns=["merchant_risk_score"])
    with pytest.raises((KeyError, ValueError)):
        get_modeling_data(broken)
