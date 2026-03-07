# FinShield Fraud Detection Dataset – Data Dictionary

This document describes the variables used in the synthetic transaction dataset.

---

## Transaction Identifiers

| Variable | Type | Description |
|--------|------|-------------|
| transaction_id | string | Unique identifier for each transaction |
| customer_id | string | Unique identifier for the customer |

---

## Time Variables

| Variable | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Date and time when the transaction occurred |
| hour | integer | Hour of the day (0–23) |
| day_of_week | integer | Day of week (0=Monday, 6=Sunday) |
| is_weekend | binary | 1 if transaction occurred on weekend |

---

## Transaction Attributes

| Variable | Type | Description |
|--------|------|-------------|
| transaction_amount | float | Amount of the transaction |
| transaction_type | category | Type of payment (card or digital payment) |
| merchant_category | category | Merchant category (grocery, electronics, travel, etc.) |
| merchant_id | string | Merchant identifier |
| merchant_risk_score | float | Synthetic risk score assigned to merchant |

---

## Channel and Device

| Variable | Type | Description |
|--------|------|-------------|
| channel | category | Transaction channel (app, web, pos) |
| device_type | category | Device used (ios, android, desktop) |
| device_id | string | Unique identifier of device |
| is_new_device | binary | Indicates whether device differs from preferred device |

---

## Location

| Variable | Type | Description |
|--------|------|-------------|
| customer_home_city | category | Customer's registered home city |
| transaction_city | category | City where the transaction occurred |
| is_foreign_transaction | binary | 1 if transaction city differs from home city |

---

## Customer Behavior

| Variable | Type | Description |
|--------|------|-------------|
| customer_tenure_days | integer | Days since customer joined |
| avg_amount_30d | float | Average transaction amount last 30 days |
| std_amount_30d | float | Standard deviation of transaction amounts |
| transactions_last_7d | integer | Number of transactions in the last 7 days |
| declines_last_30d | integer | Number of declined transactions |
| chargebacks_last_90d | integer | Number of chargebacks |

---

## Fraud Features

| Variable | Type | Description |
|--------|------|-------------|
| amount_vs_avg_ratio | float | Ratio between transaction amount and historical average |
| is_high_amount | binary | Flag indicating unusually high transaction amount |
| is_night_transaction | binary | Transaction occurred between 12am and 4am |
| is_high_risk_merchant | binary | Merchant risk score above threshold |

---

## Target

| Variable | Type | Description |
|--------|------|-------------|
| is_fraud | binary | Target variable (1 = fraudulent transaction) |
| fraud_risk_score | float | Synthetic risk score used to generate fraud label |