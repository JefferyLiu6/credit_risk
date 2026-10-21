# Credit Risk Champion–Challenger

A governance-first credit default modeling project built on the Home Credit Default Risk
dataset. Instead of declaring the highest validation score the winner, this repository keeps an
interpretable Logistic Regression incumbent, benchmarks XGBoost and LightGBM on the same frozen
split, and requires explicit promotion gates before a challenger can become champion.

## Current decision

| Model | Role | Validation ROC-AUC | Average precision | Brier score | KS |
|---|---|---:|---:|---:|---:|
| Logistic Regression | incumbent champion | 0.7293 | 0.2094 | 0.0695 | 0.3437 |
| XGBoost | leading challenger | **0.7454** | **0.2334** | **0.0684** | **0.3680** |
| LightGBM | challenger | 0.7447 | 0.2328 | 0.0684 | 0.3654 |

**Decision:** XGBoost advances to locked-test review. Logistic Regression remains champion. The
numbers above are validation results from a stratified random split—not out-of-time performance
and not evidence of production readiness.

## What makes this version different

- A frozen 60/20/20 applicant-level split is shared by all models.
- Preprocessing is fitted inside each model pipeline on training data only.
- Promotion is policy-driven: discrimination, precision-recall, calibration, and KS must all pass.
- Candidate lift is measured with a paired bootstrap on the same applications.
- Fairness and drift utilities operate at the decision and population levels.
- The API serves only the approved champion by default, not the latest experiment.
- Test-set access is a deliberate final gate, not another tuning loop.

## Repository layout

```text
credit-risk-champion-challenger/
├── src/credit_risk_champion_challenger/
│   ├── data.py          # schema validation and deterministic splits
│   ├── evaluation.py    # AUC, AP, Brier, KS, paired bootstrap
│   ├── governance.py    # candidate promotion gates
│   ├── fairness.py      # group approval and error-rate checks
│   ├── monitoring.py    # numeric/categorical PSI
│   ├── reporting.py     # reproducible validation decision report
│   ├── service.py       # framework-independent scoring service
│   └── api.py           # FastAPI adapter for the champion
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_champion_challenger.ipynb
│   ├── 04_model_validation.ipynb
│   ├── 05_credit_policy_business_value.ipynb
│   ├── 06_explainability_fairness.ipynb
│   └── 07_monitoring_stress_testing.ipynb
├── scripts/             # repeatable data preparation
├── models/              # local fitted pipelines (ignored by Git)
├── reports/             # validation evidence
├── docs/                # model card and governance checklist
└── tests/
```

## Quick start

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync
uv run pytest
uv run credit-risk validation-report
uv run uvicorn credit_risk_champion_challenger.api:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs`. The local
`models/logistic_champion.joblib` artifact must exist before the health or prediction endpoints can
serve traffic.

## Reproduce the workflow

1. Download the Home Credit competition files into `data/raw/`.
2. Run `uv run python scripts/01_prepare_data.py` once to freeze applicant assignments.
3. Execute notebooks 01–03 in order. They save the fitted pipelines and validation predictions.
4. Use Notebooks 04–07 for statistical validation, policy economics, explainability/fairness, and
   monitoring/stress testing. The committed copies include representative outputs.
5. Run `uv run credit-risk validation-report` to apply the promotion gates and bootstrap AUC lift.
6. Complete the checklist in `docs/governance_checklist.md` before opening the test set.

## Notebook portfolio story

| Notebook | Knowledge demonstrated |
|---|---|
| 01 — Data exploration | target imbalance, data quality, leakage awareness |
| 02 — Logistic Regression | interpretable baseline and pipeline-safe preprocessing |
| 03 — Champion–challenger | comparable model training and multi-metric evaluation |
| 04 — Model validation | paired bootstrap uncertainty, calibration, promotion governance |
| 05 — Credit policy | risk deciles, approval frontier, loss capture, unit economics |
| 06 — Explainability/fairness | gain importance, SHAP, decision-level group diagnostics |
| 07 — Monitoring/stress | feature and score PSI, stress scenarios, response ownership |

The large Kaggle files, fitted model binaries, and local environment are intentionally excluded
from version control. The lockfile, source, tests, reports, and governance evidence are tracked.

## Prediction request

```json
{
  "AMT_INCOME_TOTAL": 180000,
  "AMT_CREDIT": 450000,
  "AMT_ANNUITY": 24000,
  "DAYS_BIRTH": -15000,
  "DAYS_EMPLOYED": -2500,
  "EXT_SOURCE_1": 0.55,
  "EXT_SOURCE_2": 0.62,
  "EXT_SOURCE_3": 0.48,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education"
}
```

The response contains a default probability and descriptive risk band. It is a model score, not an
automatic lending decision. Threshold selection and adverse-action processes belong to the lender's
approved credit policy.

## Limitations

- The training data is a public competition dataset, not a current lender portfolio.
- The split is random and cannot demonstrate stability through time or across markets.
- Only ten application-level features are modeled; bureau and payment-history tables are excluded.
- Protected-class evaluation and local regulatory review are required before any real use.
- The repository is educational and must not be used as the sole basis for a credit decision.

See [the model card](docs/model_card.md) for intended use, known risks, and release criteria.
