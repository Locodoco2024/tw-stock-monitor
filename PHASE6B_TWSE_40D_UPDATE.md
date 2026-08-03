# Phase 6B - TWSE 40-day return-rank validation

This phase retrains the TWSE `full40_l2_1e-3` model against the same-day future
40-trading-day return percentile. It does not reuse the rejected 20-day model
weights and does not deploy notifications.

Run on Windows PowerShell:

```powershell
.\scripts\run-institutional-twse-40d-validation.ps1
```

Primary report:

```text
research/output/twse/phase6b_twse_40d_validation_reports.zip
```

The validation gate uses complete confirmation years excluding the latest year:

- at least three confirmation years;
- positive yearly spread in at least 60% of years and at least two years;
- positive top-20% minus bottom-20% 40-day adjusted return;
- positive average daily rank correlation;
- positive lower bound of the 95% monthly moving-block bootstrap interval.

Even when this gate passes, deployment remains disabled until a separate TWSE
lifecycle validation and final-model training phase is completed.
