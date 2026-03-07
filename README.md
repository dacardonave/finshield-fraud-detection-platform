# FinShield Fraud Detection Platform

## Overview
FinShield is a simulated fintech fraud detection platform designed to identify potentially fraudulent card and digital payment transactions.

This project demonstrates an end-to-end machine learning workflow, including:

- Synthetic transaction data generation
- Fraud-oriented feature engineering
- Fraud classification modeling
- Probability scoring
- API deployment for real-time inference
- Optional interactive demo with Streamlit

## Business Problem
Fintech companies process thousands of transactions daily through cards and digital payment channels. Fraud detection systems must identify suspicious activity quickly while minimizing friction for legitimate users.

The goal of this project is to predict whether a transaction is fraudulent and estimate its fraud probability.

## Objectives
- Simulate realistic financial transaction data
- Engineer behavioral and transactional fraud signals
- Train and compare fraud detection models
- Expose predictions through a FastAPI endpoint
- Build a professional portfolio project aligned with fintech and risk analytics

## Target Variable
- `is_fraud`: binary target (0 = legitimate, 1 = fraud)
- `fraud_probability`: model output probability

## Initial Scope
- Transaction types: card + digital payments
- Initial dataset size: 100,000 transactions
- Fraud prevalence target: approximately 3%
- Demo: FastAPI required, Streamlit optional

## Proposed Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Matplotlib / Plotly
- Joblib

## Project Structure
```text
finshield-fraud-detection-platform/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_simulation.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── api/
│   └── main.py
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── README.md
├── requirements.txt
└── .gitignore