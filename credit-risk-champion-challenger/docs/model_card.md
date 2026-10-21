# Model card: Credit Risk Champion–Challenger

## Summary

The system estimates the probability that a Home Credit applicant has `TARGET = 1` (payment
difficulty/default in the competition definition). Logistic Regression is the incumbent champion.
XGBoost and LightGBM are experimental challengers trained on the same features and frozen split.

## Intended use

- Education, portfolio modeling demonstrations, and champion–challenger process design.
- Ranking applicants by modeled risk for offline analysis.
- Demonstrating how model evidence can be separated from model promotion.

It is not approved for autonomous credit decisions, pricing, line assignment, collections, or
adverse-action notices.

## Data and split

- Source: Kaggle Home Credit Default Risk, main application table.
- Development split: stratified random 60% train, 20% validation, 20% locked test.
- Seed: 42.
- Unit of observation: one unique `SK_ID_CURR` applicant.
- Known limitation: the absence of an application timestamp prevents genuine out-of-time testing.

## Inputs

Eight numeric variables cover application amounts, applicant age/employment, and three external
scores. Two categorical variables represent income and education type. Missing values are imputed
inside the pipeline; categorical values unseen during training are ignored by one-hot encoding.

## Validation evidence

| Model | ROC-AUC | Average precision | Brier | KS |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7293 | 0.2094 | 0.0695 | 0.3437 |
| XGBoost | 0.7454 | 0.2334 | 0.0684 | 0.3680 |
| LightGBM | 0.7447 | 0.2328 | 0.0684 | 0.3654 |

These are validation-only point estimates. The generated validation decision report adds paired
bootstrap uncertainty for challenger-versus-champion AUC lift.

## Promotion policy

A candidate can advance to the locked test only when validation shows ROC-AUC lift of at least
0.005, no reduction in average precision or KS, and Brier score degradation no worse than 0.002.

Passing these gates is necessary but insufficient for production promotion. The candidate must also
pass the locked test, calibration review, segment/fairness review, stability checks, explainability
review, reproducibility checks, and human approval.

## Fairness and human oversight

The project exposes approval-rate, default-catch, and good-customer false-decline gaps by group.
Protected attributes should be used for auditing, not as model inputs. A metric threshold is an alert
for investigation and legal/compliance review, not an automatic finding of discrimination. Human
review is required for model promotion and for any policy that maps score to decision.

## Monitoring

Numeric and categorical PSI are provided with three operational bands: below 0.10 stable, 0.10–0.25
review, and 0.25 or above material. Monitor model inputs and score distributions. PSI signals
population movement; it does not prove performance degradation when outcomes are unavailable.

## Known limitations and risks

- Public historical data may not represent a particular institution or current applicants.
- Random validation can overstate generalization under economic or policy shifts.
- External-score provenance and availability must be reviewed before deployment.
- The score is correlational, not causal, and must not be presented as a reason for default.
- Risk bands are descriptive. They are not approved decision thresholds.

## Current status

Logistic Regression remains the approved project champion. XGBoost is the leading validation
candidate and may advance to a single locked-test evaluation after the governance checklist is
signed off.

