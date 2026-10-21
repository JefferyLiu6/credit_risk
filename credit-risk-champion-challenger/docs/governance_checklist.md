# Champion–challenger governance checklist

## Before locked-test access

- [x] Applicant IDs are unique and labels are binary and complete.
- [x] The 60/20/20 split is persisted and reused across models.
- [x] Preprocessing is learned on training data only.
- [x] Champion and challengers use the same ten model inputs.
- [x] Validation metrics include ROC-AUC, average precision, Brier score, and KS.
- [x] Promotion thresholds are documented before test evaluation.
- [ ] Paired bootstrap confidence interval reviewed and saved.
- [ ] Candidate calibration plot and segment diagnostics reviewed.
- [ ] Fairness audit completed for available protected-group proxies.
- [ ] Challenger explanation review completed.
- [ ] Independent reviewer authorizes one-time test access.

## Locked-test gate

- [ ] Run only the frozen incumbent and selected candidate.
- [ ] Do not tune features, hyperparameters, calibration, or thresholds after viewing test results.
- [ ] Record point estimates, confidence intervals, sample size, code version, and artifact hashes.
- [ ] Confirm candidate meets the pre-registered gates on the locked test.
- [ ] Confirm calibration and material portfolio segments have no unexplained regression.

## Release gate

- [ ] Model card updated with locked-test evidence.
- [ ] Data/model lineage and reproducible environment recorded.
- [ ] Drift reference population and alert ownership established.
- [ ] Rollback artifact and rollback procedure tested.
- [ ] Lending-policy thresholds approved separately from model ranking.
- [ ] Compliance, model-risk, and business owners sign off.

Until every required item is complete, `logistic_champion.joblib` remains the served model.

