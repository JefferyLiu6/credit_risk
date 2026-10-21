# Credit Risk Modeling & ML Validation

**A Python portfolio project exploring how machine learning can support better lending decisions.**

Using **307,511 loan applications**, this project compares three models that estimate default risk, evaluates their reliability, and demonstrates how their scores could inform lending policy. It combines seven documented notebooks with reusable Python modules, a local prediction API, automated tests, and a CI workflow.

[Explore the full project](credit-risk-champion-challenger/README.md) · [View model comparison](credit-risk-champion-challenger/notebook/03_champion_challenger.ipynb) · [View business analysis](credit-risk-champion-challenger/notebook/05_credit_policy_business_value.ipynb)

## Project at a glance

| Area | What the project demonstrates |
|---|---|
| Business problem | Estimate default risk and explore the tradeoff between approving more applicants and accepting more credit losses |
| Machine learning | Compare Logistic Regression, XGBoost, and LightGBM on the same applicants and features |
| Main result | XGBoost achieved **0.7454 validation ROC-AUC**, compared with **0.7293** for Logistic Regression |
| Model reliability | Assess uncertainty, probability calibration, explanations, group differences, and simulated population changes |
| Software engineering | Package preprocessing and scoring, expose a FastAPI endpoint, and maintain tests and a GitHub Actions workflow |
| Communication | Explain assumptions, findings, decisions, and limitations in seven executed notebooks |

ROC-AUC measures how well a model ranks applicants by risk: 0.5 represents random ranking and 1.0 represents perfect separation. The improvement here is **0.0161 AUC points** on validation data.

## Skills demonstrated

- **Applied machine learning:** classification with an imbalanced target, reproducible splits, preprocessing, and model comparison.
- **Statistical evaluation:** paired bootstrap confidence intervals, calibration checks, and multiple performance metrics.
- **Business reasoning:** risk segmentation, approval-rate analysis, and illustrative lending economics.
- **Responsible AI:** SHAP explanations, group fairness diagnostics, documented assumptions, and model review criteria.
- **Python engineering:** reusable modules, serialized pipelines, API input validation, unit tests, and continuous integration configuration.

**Tools:** Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · pytest · GitHub Actions

## Suggested review path

1. [Model comparison](credit-risk-champion-challenger/notebook/03_champion_challenger.ipynb) — see how the models perform and why the strongest candidate was selected for further review.
2. [Credit policy and business value](credit-risk-champion-challenger/notebook/05_credit_policy_business_value.ipynb) — see how model outputs connect to lending decisions.
3. [Model validation](credit-risk-champion-challenger/notebook/04_model_validation.ipynb) — examine uncertainty and probability quality.
4. [Python implementation](credit-risk-champion-challenger/src/credit_risk_champion_challenger/) — inspect the reusable components behind the analysis.

## Scope and project status

This is an educational portfolio project using the public Home Credit dataset. Results come from a stratified random validation split; they do not establish performance on future applications. Logistic Regression remains the project's baseline champion, while XGBoost is the leading candidate for further evaluation.

The repository includes a local scoring API and simulated monitoring exercises. Cloud deployment and live lending operations are outside the implemented scope.

For the complete notebook sequence, results, and setup instructions, open the [project README](credit-risk-champion-challenger/README.md).
