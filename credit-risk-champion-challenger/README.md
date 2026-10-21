# Credit Risk Champion–Challenger

**From loan application data to model evaluation, business decisions, and a working scoring prototype.**

This project investigates a practical question: **does a more complex machine learning model improve credit-risk predictions enough to justify its added complexity?**

It uses **307,511 Home Credit applications** to compare an interpretable Logistic Regression baseline with XGBoost and LightGBM. The analysis then examines statistical reliability, lending-policy tradeoffs, explainability, group outcomes, and model behavior under simulated changes in the applicant population.

“Champion–challenger” means comparing an existing baseline model—the champion—with alternative models before deciding whether to replace it. Here, it is a portfolio demonstration of that process.

[Model comparison](notebook/03_champion_challenger.ipynb) · [Business analysis](notebook/05_credit_policy_business_value.ipynb) · [Source code](src/credit_risk_champion_challenger/) · [Model card](docs/model_card.md)

## Key outcomes

- **Compared three models fairly:** each uses the same ten input features and saved training/validation assignments.
- **Improved validation ranking:** XGBoost achieved **0.7454 ROC-AUC**, compared with **0.7293** for the baseline, and improved average precision and probability error.
- **Evaluated reliability:** paired bootstrap comparisons, calibration diagnostics, SHAP explanations, and group audits extend the analysis beyond a single score.
- **Connected predictions to business decisions:** explored approval rates, default capture, and contribution under explicit illustrative lending assumptions.
- **Built supporting software:** reusable Python modules, complete preprocessing/model pipelines, a FastAPI scoring endpoint, unit tests, and a GitHub Actions testing workflow.

All reported model results below are **validation results**. XGBoost is the leading candidate for further review; the project retains Logistic Regression as its baseline champion.

## Model results

| Model | ROC-AUC ↑ | Average precision ↑ | Brier score ↓ | KS ↑ |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7293 | 0.2094 | 0.0695 | 0.3437 |
| **XGBoost** | **0.7454** | **0.2334** | **0.0684** | **0.3680** |
| LightGBM | 0.7447 | 0.2328 | 0.0684 | 0.3654 |

- **ROC-AUC:** how well the model ranks applicants by default risk.
- **Average precision:** how effectively it identifies defaults when they are relatively uncommon.
- **Brier score:** error in predicted probabilities; lower is better.
- **KS:** the maximum separation between defaulter and non-defaulter score distributions.

XGBoost improves ROC-AUC by **0.0161 points** over the baseline. Paired bootstrap analysis examines the uncertainty in this improvement. The comparison supports further evaluation, while final model selection also depends on calibration, segment behavior, stability, and review.

Evidence: [saved validation metrics](reports/notebook3_validation_metrics.csv), [experiment metadata](reports/notebook3_metadata.json), and [validation notebook](notebook/04_model_validation.ipynb).

## What to explore

The notebooks include executed results, charts, methodological explanations, and limitations. They can be reviewed directly on GitHub.

| Notebook | Question answered | Skills demonstrated |
|---|---|---|
| [01 — Data exploration](notebook/01_data_exploration.ipynb) | What data-quality issues and risk patterns need attention? | Missingness, class imbalance, anomalous values, segment analysis |
| [02 — Logistic Regression](notebook/02_logistic_regression.ipynb) | What can an interpretable baseline achieve? | Training-only preprocessing, coefficient interpretation, calibration |
| [03 — Champion–challenger comparison](notebook/03_champion_challenger.ipynb) | Do more complex models improve the baseline? | Controlled comparison, boosting, multi-metric evaluation |
| [04 — Model validation](notebook/04_model_validation.ipynb) | How reliable is the observed improvement? | Paired bootstrap, probability quality, review criteria |
| [05 — Credit policy and business value](notebook/05_credit_policy_business_value.ipynb) | How do approval rates affect risk and illustrative contribution? | Risk deciles, default capture, policy tradeoffs, unit economics |
| [06 — Explainability and fairness](notebook/06_explainability_fairness.ipynb) | What drives the model, and how do outcomes differ across groups? | SHAP, feature importance, group approval and error rates |
| [07 — Monitoring and stress testing](notebook/07_monitoring_stress_testing.ipynb) | How does the model respond when the applicant population changes? | Feature/score drift, scenario analysis, monitoring response design |

**Short on time?** Start with Notebooks 03 and 05 for the result and business context, then review Notebook 04 for statistical reasoning.

## Design decisions and tradeoffs

**A compact baseline.** Eight numeric and two categorical application features keep the comparison understandable. Bureau and repayment-history tables are available in the source dataset but are not used by these models.

**Consistent development data.** A saved, stratified 60% training / 20% validation / 20% test split preserves the applicant assignments across experiments. Model performance reported here is measured on validation.

**Preprocessing travels with the model.** Employment sentinel normalization, imputation, scaling, and encoding are included in the saved pipelines. Learned preprocessing parameters are fitted on training data.

**Several criteria inform the decision.** ROC-AUC is considered alongside average precision, probability error, KS, explanations, and group diagnostics. The implemented comparison gates support review; they do not constitute a production approval system.

**Business assumptions stay visible.** Notebook 05 uses hypothetical margins, exposure, and loss severity. Its contribution estimates demonstrate decision analysis and are not realized savings or revenue.

## Engineering implementation

| Component | Implementation |
|---|---|
| Data preparation | [Schema checks and split helpers](src/credit_risk_champion_challenger/data.py) |
| Evaluation | [Performance metrics and paired bootstrap](src/credit_risk_champion_challenger/evaluation.py) |
| Model comparison rules | [Candidate review gates](src/credit_risk_champion_challenger/governance.py) |
| Group diagnostics | [Approval and error-rate analysis](src/credit_risk_champion_challenger/fairness.py) |
| Drift analysis | [Numeric and categorical PSI](src/credit_risk_champion_challenger/monitoring.py) |
| Inference | [Scoring service](src/credit_risk_champion_challenger/service.py) and [FastAPI endpoints](src/credit_risk_champion_challenger/api.py) |
| Verification | [Unit tests](tests/) and [GitHub Actions workflow](../.github/workflows/tests.yml) |
| Documentation | [Model card](docs/model_card.md) and [review checklist](docs/governance_checklist.md) |

**Stack:** Python 3.12+ · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · Matplotlib · FastAPI · pytest · Ruff · uv

## Run the project locally

From the repository root:

```bash
cd credit-risk-champion-challenger
uv sync
uv run pytest
```

To rebuild the decision report from the included validation predictions:

```bash
uv run credit-risk validation-report
```

To reproduce model training and the full notebook sequence:

1. Obtain the Home Credit Default Risk dataset and place `application_train.csv` in `data/raw/`.
2. Run `uv run python scripts/01_prepare_data.py` once to create the split assignments.
3. Run Notebooks 01–03 in order using the project's Python environment. These produce the model files and validation predictions.
4. Run Notebooks 04–07 to reproduce the validation, policy, explanation, and monitoring analyses.

Raw datasets, fitted model binaries, and the local environment are excluded from Git. Recreate the models through the notebooks before running the scoring API.

### Local prediction API

After training:

```bash
uv run uvicorn credit_risk_champion_challenger.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the endpoint. The service provides `GET /health` and `POST /predict`, using the saved Logistic Regression pipeline by default.

Example request:

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

The response contains an estimated default probability, a descriptive risk band, and the model name.

## Scope and next steps

This is a portfolio project using historical competition data. The random split does not establish performance through time, the fairness analysis is exploratory, and the stress scenarios are synthetic. The API is a local prototype; deployment, security controls, live alerts, and model registry integration are not implemented.

Further development could add SQL-based relational features, automated training, API integration tests, containerization, cloud deployment, and experiment tracking. Time-based validation would require suitable dated data.

See the [model card](docs/model_card.md) for intended use and limitations.
