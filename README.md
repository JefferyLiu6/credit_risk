# Credit Risk

This repository contains my governance-first take on an end-to-end credit-risk ML project.
The implementation lives in
[`credit-risk-champion-challenger`](credit-risk-champion-challenger/README.md), where an
interpretable Logistic Regression incumbent is compared with XGBoost and LightGBM under explicit
promotion, fairness, stability, and locked-test controls.

## Current status

- Logistic Regression remains the incumbent champion.
- XGBoost is the leading validation challenger.
- The complete package, notebooks, reports, tests, API, model card, and governance checklist are in
  the project directory.

```bash
cd credit-risk-champion-challenger
uv sync
uv run pytest
```

