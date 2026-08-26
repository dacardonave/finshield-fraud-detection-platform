"""
streamlit_app.py

Interactive demo for the FinShield Fraud Detection Platform.

What this is
------------
A form where you fill in (or load a preset for) a single transaction and
see, live, what the trained model decides: the fraud probability, the
flag/no-flag decision, and *why* - which signals pushed the score up or
down.

How it relates to the rest of the project
------------------------------------------
This app calls `src.predict.predict_one` directly - the exact same
function the FastAPI service (`api/main.py`) calls. There is no separate
"demo" copy of the scoring logic; if you change how features are
computed or which model is deployed, this app and the API pick it up
automatically the next time `python -m src.train` is run.

Running it locally
-------------------
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import create_features
from src.predict import FEATURE_COLUMNS, load_model, predict_one

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinShield — Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FinShield — Fraud Detection Demo")
st.caption(
    "Fill in a transaction (or load a preset below) and score it against the "
    "trained model in real time. See `outputs/reports/final_report.md` for "
    "the full model comparison and the business case behind the decision "
    "threshold used here."
)

CATEGORIES = ["grocery", "electronics", "travel", "fashion", "gaming", "restaurants", "fuel", "utilities"]
CITIES = ["New York", "Newark", "Jersey City", "Brooklyn", "Queens", "Miami", "Los Angeles"]
DEVICES = ["ios", "android", "desktop"]

# ---------------------------------------------------------------------------
# Presets — so a reviewer can see a result in one click, no typing required
# ---------------------------------------------------------------------------

PRESETS = {
    "Typical legitimate transaction": dict(
        transaction_type="card", merchant_category="grocery", transaction_amount=45.0,
        channel="pos", device_type="ios", transaction_city="Brooklyn", merchant_risk_score=0.15,
        txn_date=date(2025, 6, 15), txn_time=time(14, 0),
        customer_home_city="Brooklyn", customer_tenure_days=900, account_age_days=900,
        preferred_device_type="ios", avg_amount_30d=50.0, std_amount_30d=20.0,
        transactions_last_7d=5, declines_last_30d=0, chargebacks_last_90d=0,
    ),
    "High-risk transaction": dict(
        transaction_type="digital_payment", merchant_category="electronics", transaction_amount=900.0,
        channel="web", device_type="android", transaction_city="Miami", merchant_risk_score=0.9,
        txn_date=date(2025, 6, 15), txn_time=time(2, 30),
        customer_home_city="Brooklyn", customer_tenure_days=900, account_age_days=900,
        preferred_device_type="ios", avg_amount_30d=50.0, std_amount_30d=20.0,
        transactions_last_7d=5, declines_last_30d=0, chargebacks_last_90d=0,
    ),
}

preset_name = st.selectbox("Quick example (optional)", ["Custom"] + list(PRESETS.keys()))
defaults = PRESETS.get(preset_name, PRESETS["Typical legitimate transaction"])

# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Transaction")
        transaction_type = st.selectbox("Transaction type", ["card", "digital_payment"],
                                         index=["card", "digital_payment"].index(defaults["transaction_type"]))
        merchant_category = st.selectbox("Merchant category", CATEGORIES,
                                          index=CATEGORIES.index(defaults["merchant_category"]))
        transaction_amount = st.number_input("Transaction amount ($)", min_value=0.01,
                                              value=defaults["transaction_amount"])
        channel = st.selectbox("Channel", ["app", "web", "pos"],
                                index=["app", "web", "pos"].index(defaults["channel"]))
        device_type = st.selectbox("Device used", DEVICES, index=DEVICES.index(defaults["device_type"]))
        transaction_city = st.selectbox("Transaction city", CITIES,
                                         index=CITIES.index(defaults["transaction_city"]) if defaults["transaction_city"] in CITIES else 0)
        merchant_risk_score = st.slider("Merchant risk score", 0.0, 1.0, defaults["merchant_risk_score"])
        txn_date = st.date_input("Date", value=defaults["txn_date"])
        txn_time = st.time_input("Time", value=defaults["txn_time"])

    with col2:
        st.subheader("Customer profile")
        customer_home_city = st.selectbox(
            "Customer home city", CITIES,
            index=CITIES.index(defaults["customer_home_city"]) if defaults["customer_home_city"] in CITIES else 0,
        )
        customer_tenure_days = st.number_input("Customer tenure (days)", min_value=0,
                                                 value=defaults["customer_tenure_days"])
        account_age_days = st.number_input("Account age (days)", min_value=0,
                                            value=defaults["account_age_days"])
        preferred_device_type = st.selectbox("Customer's usual device", DEVICES,
                                              index=DEVICES.index(defaults["preferred_device_type"]))

    with col3:
        st.subheader("Recent behavior (last 30/7/90 days)")
        avg_amount_30d = st.number_input("Avg. amount, last 30d ($)", min_value=0.0,
                                          value=defaults["avg_amount_30d"])
        std_amount_30d = st.number_input("Std. dev. of amount, last 30d ($)", min_value=0.0,
                                          value=defaults["std_amount_30d"])
        transactions_last_7d = st.number_input("Transactions, last 7d", min_value=0,
                                                 value=defaults["transactions_last_7d"])
        declines_last_30d = st.number_input("Declines, last 30d", min_value=0,
                                              value=defaults["declines_last_30d"])
        chargebacks_last_90d = st.number_input("Chargebacks, last 90d", min_value=0,
                                                 value=defaults["chargebacks_last_90d"])

    submitted = st.form_submit_button("Score this transaction", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Human-friendly labels for the explanation chart
# ---------------------------------------------------------------------------

FRIENDLY_NAMES = {
    "is_high_amount": "Amount much higher than usual for this customer",
    "is_very_high_amount": "Amount extremely higher than usual for this customer",
    "is_night_transaction": "Happened at night (12am-5am)",
    "is_high_risk_merchant": "Merchant has a high risk score",
    "is_new_device": "Device different from the customer's usual one",
    "is_foreign_transaction": "City different from the customer's home city",
    "channel_risk": "Risk level of the channel used",
    "category_risk": "Risk level of the merchant category",
    "high_amount_new_device": "High amount + new device combined",
    "foreign_high_risk_merchant": "Foreign transaction + high-risk merchant combined",
    "web_night_transaction": "Web transaction at night",
    "very_high_amount_foreign": "Very high amount + foreign transaction combined",
    "log_transaction_amount": "Transaction amount",
    "txn_velocity": "Recent transaction velocity",
    "customer_risk_score": "History of declines/chargebacks",
    "decline_ratio": "Ratio of recent declines to activity",
    "amount_merchant_risk": "Amount combined with merchant risk",
    "is_new_customer": "Customer is relatively new (<90 days)",
    "abnormal_activity": "Unusually high recent activity + spending",
    "transaction_amount": "Transaction amount",
    "merchant_risk_score": "Merchant risk score",
    "amount_vs_avg_ratio": "Amount vs. this customer's average amount",
    "avg_amount_30d": "Customer's average amount, last 30 days",
    "std_amount_30d": "Customer's amount variability, last 30 days",
    "account_age_days": "Account age",
    "customer_tenure_days": "Customer tenure",
    "transactions_last_7d": "Number of transactions, last 7 days",
    "declines_last_30d": "Number of declines, last 30 days",
    "chargebacks_last_90d": "Number of chargebacks, last 90 days",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "is_weekend": "Happened on a weekend",
}


def humanize_feature_name(name: str) -> str:
    base = name.split("__", 1)[-1]
    return FRIENDLY_NAMES.get(base, base.replace("_", " ").title())


def explain_prediction(pipeline, X_row: pd.DataFrame, top_n: int = 8):
    """
    For a linear model, each feature's contribution to the log-odds of
    fraud is simply coefficient x (preprocessed) feature value. This is
    exact (not an approximation like SHAP would need for a non-linear
    model), and it is only meaningful because the winning model here is
    Logistic Regression. Returns None for any other model type.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    X_transformed = preprocessor.transform(X_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    contributions = model.coef_.ravel() * X_transformed[0]
    series = pd.Series(contributions, index=feature_names)
    top = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)
    return top.iloc[::-1]  # smallest-magnitude first, for a nicer horizontal bar chart


# ---------------------------------------------------------------------------
# Score & display results
# ---------------------------------------------------------------------------

if submitted:
    txn_timestamp = datetime.combine(txn_date, txn_time)

    payload = dict(
        transaction_type=transaction_type,
        merchant_category=merchant_category,
        transaction_amount=transaction_amount,
        channel=channel,
        device_type=device_type,
        transaction_city=transaction_city,
        merchant_risk_score=merchant_risk_score,
        timestamp=txn_timestamp.isoformat(),
        customer_home_city=customer_home_city,
        customer_tenure_days=customer_tenure_days,
        account_age_days=account_age_days,
        preferred_device_type=preferred_device_type,
        avg_amount_30d=avg_amount_30d,
        std_amount_30d=std_amount_30d,
        transactions_last_7d=transactions_last_7d,
        declines_last_30d=declines_last_30d,
        chargebacks_last_90d=chargebacks_last_90d,
    )

    try:
        with st.spinner("Scoring transaction..."):
            result = predict_one(payload)
    except FileNotFoundError:
        st.error(
            "No trained model found at `models/model.joblib`. Run "
            "`python -m src.train` first, then reload this page."
        )
        st.stop()
    except ValueError as exc:
        st.error(f"Could not score this transaction: {exc}")
        st.stop()

    st.divider()
    st.header("Result")

    proba = result["fraud_probability"]
    threshold = result["decision_threshold"]
    is_fraud = result["is_fraud"]
    risk_tier = result["risk_tier"]

    left, right = st.columns([1, 1.3])

    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            title={"text": "Fraud probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, threshold * 50], "color": "#c6efce"},
                    {"range": [threshold * 50, threshold * 100], "color": "#ffeb9c"},
                    {"range": [threshold * 100, 100], "color": "#ffc7ce"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        ))
        gauge.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(f"Red line = decision threshold ({threshold:.0%}), chosen in Phase 2 to minimize expected business cost, not a default 50%.")

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Decision", "🚩 FLAGGED" if is_fraud else "✅ Approved")
        m2.metric("Risk tier", risk_tier.upper())
        m3.metric("Probability", f"{proba:.1%}")

        X_row = pd.DataFrame([payload])
        X_row["timestamp"] = pd.to_datetime(X_row["timestamp"])
        X_row["hour"] = X_row["timestamp"].dt.hour
        X_row["day_of_week"] = X_row["timestamp"].dt.dayofweek
        X_row["is_weekend"] = X_row["day_of_week"].isin([5, 6]).astype(int)

        featurized = create_features(X_row)[FEATURE_COLUMNS]
        contributions = explain_prediction(load_model(), featurized)

        if contributions is not None:
            st.subheader("What drove this score")
            colors = ["#e15759" if v > 0 else "#4e79a7" for v in contributions.values]
            fig = go.Figure(go.Bar(
                x=contributions.values,
                y=[humanize_feature_name(n) for n in contributions.index],
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Contribution to fraud log-odds (red = increases risk, blue = decreases it)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Per-prediction explanation is only implemented for linear models.")

    with st.expander("Raw prediction payload"):
        st.json(result)
