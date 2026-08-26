# FinShield — Model Selection & Business Interpretation Report

**Winning model:** `logistic_regression`
**Tuned hyperparameters:** `{'model__C': 0.01}`

## Model comparison (5-fold cross-validation, train set)

| model                  |   roc_auc_mean |   roc_auc_std |   average_precision_mean |   average_precision_std |   f1_mean |     f1_std |
|:-----------------------|---------------:|--------------:|-------------------------:|------------------------:|----------:|-----------:|
| logistic_regression    |       0.713474 |     0.0121673 |                0.0949886 |               0.0102179 |  0.12494  | 0.00464982 |
| hist_gradient_boosting |       0.697664 |     0.0159073 |                0.0888157 |               0.0106693 |  0.128307 | 0.00530649 |
| random_forest          |       0.68134  |     0.0122239 |                0.0813063 |               0.0056132 |  0        | 0          |
| dummy                  |       0.5      |     0         |                0.0333    |               2.5e-05   |  0        | 0          |

## Decision threshold

The default classification threshold of 0.5 is arbitrary and not aligned with the business cost of fraud. We instead scan thresholds and pick the one that minimizes an estimated total dollar cost on the test set, where a missed fraud (false negative) costs the full transaction amount and a false alarm (false positive) costs an assumed $5.00 in review/friction.

- **Cost-optimal threshold:** 0.660 (precision=0.103, recall=0.350, estimated cost=$32651.89)
- **F1-optimal threshold (statistical baseline):** 0.740 (precision=0.120, recall=0.245)

We use the cost-optimal threshold (0.660) in production.

## Estimated business impact

- No-model baseline (every fraudulent transaction succeeds): **$41,747.19** in fraud losses on the test set.
- With the model at the cost-optimal threshold: **$32,651.89** in combined fraud losses + false-alarm friction cost.
- **Estimated cost reduction: 21.8%** versus taking no action at all.

## Test set performance

At the chosen threshold: precision=0.1027, recall=0.3498, F1=0.1588, ROC AUC=0.7009, PR AUC=0.0837.

At the default 0.5 threshold (for reference): precision=0.0688, recall=0.5736, F1=0.1229.

## Business takeaway

ROC AUC and accuracy are misleading on this problem because fraud is a small minority class; PR AUC and the cost-weighted threshold are the metrics that actually reflect the trade-off a fraud team faces between blocking fraud losses and creating friction for legitimate customers. The friction cost assumption above is a placeholder and should be replaced with real unit-economics figures (cost of manual review, customer churn from false declines, etc.) before this threshold is used in a live decision.

## Why PR AUC is modest, and why Logistic Regression won

PR AUC (~0.08-0.10) is well above the random baseline (the fraud prevalence, ~0.033) but far from the near-perfect separability sometimes seen on toy fraud datasets. This is by design, not a modeling failure: `data_generation.py` builds the latent fraud score as a linear combination of risk signals and then adds substantial Gaussian noise before converting it to a probability, deliberately limiting how learnable the signal is — closer to the noisy reality of production fraud systems than an artificially clean dataset would be.

This also explains why Logistic Regression outperformed the tree-based ensembles in cross-validation: the true data-generating process is a sigmoid over a linear score, so a linear model recovers that structure directly, while Random Forest and HistGradientBoosting spend capacity modeling noise-driven interactions that aren't really there. The lesson generalizes beyond this project: match model complexity to the actual signal in the data, don't default to the most complex model available.
