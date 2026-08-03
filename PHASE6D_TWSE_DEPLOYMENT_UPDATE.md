# Phase 6D — TWSE final model and production deployment

## Fixed production specification

- Market: TWSE listed common stocks
- Model target: same-day cross-sectional future 40-market-day return rank
- Feature set: market-specific `full40`
- Daily ranking universe:
  - median trading amount over 20 market days >= NT$100,000,000
  - median trading volume over 20 market days >= 300 board lots
  - at least 18 normal trading days
  - no 3-day zero-volume streak
- Entry event: model percentile >= 90 for 3 consecutive market days
- Tracking period: 40 market days
- Cooldown after event end: 20 market days
- Action: `TRACK_ONLY`

The TWSE model, rolling state, notification plan, event IDs and Discord labels are
separate from TPEx. Existing TPEx model files and `runtime/institutional` state are
not overwritten.

## Local execution order

```powershell
python -m pytest -q
python -m compileall -q src research tests scripts
.\scripts\run-institutional-twse-final.ps1
.\scripts\seed-twse-institutional-deployment.ps1
.\scripts\publish-twse-institutional-seed.ps1
```

Commit the generated model files:

```text
models/twse/phase5d_final_rank_model.json
models/twse/phase5d_final_rank_model.npz
```

Research reports under `research/output/twse` remain local because that directory
is ignored by Git.

## GitHub Actions

`TWSE institutional daily update` runs at 18:10 Taiwan time on weekdays and can
also be run manually with an optional `as_of_date`. The shared state workflow
preserves both:

```text
institutional/
institutional_twse/
state.json
```

The hourly monitor reads both notification plans when institutional candidates
are enabled. `max_new_candidates` remains one combined per-user limit across both
markets for that monitor run.

## Notification interpretation

TWSE notifications explicitly show `TWSE 上市` and `同日上市法人排名`. They include
the same 20-day three-institution estimated cost reference as TPEx. The estimated
cost is not a real inventory cost and does not create buy, sell or chase-risk
advice.
