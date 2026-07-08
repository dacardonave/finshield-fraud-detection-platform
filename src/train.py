"""
train.py

Training script for the FinShield Fraud Detection Platform.
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.feature_engineering import get_modeling_data


RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = "data/raw/transactions.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(categorical_cols: list[str], numerical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def evaluate_model(model_name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print(f"{'=' * 60}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nROC AUC:", round(metrics["roc_auc"], 4))
    print("PR AUC:", round(metrics["pr_auc"], 4))

    return metrics


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    X, y, entity_df, categorical_cols, numerical_cols = get_modeling_data(df)

    X_train, X_test, y_train, y_test, entity_train, entity_test = train_test_split(
        X,
        y,
        entity_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train fraud rate:", round(y_train.mean(), 4))
    print("Test fraud rate:", round(y_test.mean(), 4))

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    all_metrics = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        all_metrics[model_name] = metrics

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"\nSaved model to: {model_path}")

        # Save scored predictions with identifiers for traceability
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        scored_df = entity_test.copy()
        scored_df["y_true"] = y_test.values
        scored_df["y_pred"] = y_pred
        scored_df["y_proba"] = y_proba

        scored_path = MODEL_DIR / f"{model_name}_test_predictions.csv"
        scored_df.to_csv(scored_path, index=False)
        print(f"Saved test predictions to: {scored_path}")

    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()