# FinShield Fraud Detection Platform – Data Dictionary

This document describes the variables used in the synthetic fraud detection dataset for the FinShield project.

---

## 1. Target Variable

| Variable | Type | Description |
|---|---|---|
| is_fraud | binary | Target variable indicating whether the transaction is fraudulent (1) or legitimate (0). |

---

## 2. Identifier Variables

These variables identify entities but are not intended for model training.

| Variable | Type | Description |
|---|---|---|
| transaction_id | string | Unique identifier for each transaction. |
| customer_id | string | Unique identifier for each customer. |
| merchant_id | string | Unique identifier for each merchant. |
| device_id | string | Unique identifier for the device used in the transaction. |

---

## 3. Time Variables

| Variable | Type | Description |
|---|---|---|
| timestamp | datetime | Date and time of the transaction. |
| hour | integer | Hour of the transaction (0–23). |
| day_of_week | integer | Day of the week (0 = Monday, 6 = Sunday). |
| is_weekend | binary | Indicates whether the transaction occurred on a weekend. |

---

## 4. Transaction Variables

| Variable | Type | Description |
|---|---|---|
| transaction_type | category | Type of payment method used (e.g. card or digital payment). |
| merchant_category | category | Category of merchant where the transaction occurred. |
| transaction_amount | float | Monetary value of the transaction. |
| channel | category | Transaction channel: app, web, or POS (point of sale). |
| transaction_city | category | City where the transaction took place. |
| merchant_risk_score | float | Synthetic merchant-level risk score based on category and simulated risk behavior. |

---

## 5. Customer Profile Variables

| Variable | Type | Description |
|---|---|---|
| customer_home_city | category | Registered home city of the customer. |
| customer_tenure_days | integer | Number of days since the customer joined the platform. |
| account_age_days | integer | Number of days since the account was created. |
| preferred_device_type | category | Customer’s usual or preferred device type. |
| avg_amount_30d | float | Historical average transaction amount over the last 30 days. |
| std_amount_30d | float | Historical standard deviation of transaction amounts over the last 30 days. |
| transactions_last_7d | integer | Number of customer transactions in the last 7 days. |
| declines_last_30d | integer | Number of declined transactions in the last 30 days. |
| chargebacks_last_90d | integer | Number of chargebacks in the last 90 days. |

---

## 6. Device Variables

| Variable | Type | Description |
|---|---|---|
| device_type | category | Device used in the transaction (e.g. iOS, Android, desktop). |

---

## 7. Simulation-Based Fraud Signals

These variables were generated during the simulation process to create realistic fraud patterns.

| Variable | Type | Description |
|---|---|---|
| amount_vs_avg_ratio | float | Ratio between current transaction amount and the customer’s historical average amount. |
| is_high_amount | binary | Indicates whether the amount is unusually high compared with the customer’s average behavior. |
| is_very_high_amount | binary | Indicates whether the amount is extremely high compared with the customer’s average behavior. |
| is_night_transaction | binary | Indicates whether the transaction occurred during night hours. |
| is_high_risk_merchant | binary | Indicates whether the merchant risk score is above the defined threshold. |
| is_new_device | binary | Indicates whether the device used differs from the customer’s preferred device type. |
| is_foreign_transaction | binary | Indicates whether the transaction occurred outside the customer’s home city. |
| channel_risk | float | Synthetic base risk associated with the transaction channel. |
| category_risk | float | Synthetic base risk associated with the merchant category. |
| high_amount_new_device | binary | Interaction flag for high amount and new device. |
| foreign_high_risk_merchant | binary | Interaction flag for foreign transaction and high-risk merchant. |
| web_night_transaction | binary | Interaction flag for web transaction during night hours. |
| very_high_amount_foreign | binary | Interaction flag for very high amount and foreign transaction. |
| fraud_risk_score | float | Synthetic latent fraud score used internally during simulation. |
| fraud_probability | float | Simulated fraud probability derived from the latent fraud score. |

---

## 8. Additional Engineered Features

These features were created to strengthen the modeling dataset and better capture behavioral fraud patterns.

| Variable | Type | Description |
|---|---|---|
| log_transaction_amount | float | Log-transformed transaction amount used to reduce skewness and stabilize the heavy-tailed distribution of monetary values. |
| txn_velocity | float | Average number of transactions per day over the last 7 days. Captures recent transaction intensity. |
| customer_risk_score | float | Composite customer-level risk score based on prior declines and chargebacks. |
| decline_ratio | float | Ratio of declined transactions to recent transaction activity. Includes a smoothing term in the denominator to avoid division by zero. |
| amount_merchant_risk | float | Interaction between transaction amount and merchant risk score. High-value transactions at risky merchants are more suspicious. |
| is_new_customer | binary | Indicates whether the customer is relatively new to the platform (e.g. tenure below 90 days). |
| abnormal_activity | binary | Indicates unusually high recent activity combined with spending above the customer’s normal behavior. |

---

## 9. Variables Excluded from Model Training

These variables are excluded from the predictive model because they represent identifiers, raw timestamps, or direct information leakage.

| Variable | Reason for Exclusion |
|---|---|
| transaction_id | Identifier only; no predictive business meaning. |
| customer_id | Identifier only; may introduce artificial memorization. |
| merchant_id | Identifier only; high-cardinality field not yet engineered. |
| device_id | Identifier only; high-cardinality field not yet engineered. |
| timestamp | Raw datetime field; excluded in favor of derived time-based variables. |
| fraud_risk_score | Direct simulation artifact that leaks target-generation logic. |
| fraud_probability | Direct simulation artifact that leaks target-generation logic. |

---

## 10. Notes

- The dataset is synthetic but designed to mimic realistic fraud patterns in fintech transactions.
- Several variables are behavior-based and intentionally structured to reflect common fraud signals such as abnormal spending, risky channels, suspicious merchants, and device inconsistency.
- Additional temporal features may be introduced in future iterations of the project.