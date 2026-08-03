# Phase 6C — TWSE 40-day lifecycle and liquidity validation

Phase 6C consumes the frozen Phase 6B out-of-sample scores. It does not retrain
or retune the TWSE model.

## Fixed rule grid

- Entry percentile: top 15%, top 10%, top 5%.
- Consecutive confirmation: 1, 3, 5, 10 market days.
- Cooldown comparison: 0, 10, 20, 40 market days.
- Primary outcome: 40-market-day adjusted return and same-day-universe excess return.
- Extension outcomes: 60 and 80 market days.

## Liquidity universes

Each universe rebuilds the same-day percentile after filtering:

- Median 20-day trading money >= NT$20 million.
- Median 20-day trading money >= NT$50 million.
- Median 20-day trading money >= NT$100 million.
- NT$50 million plus median 20-day volume >= 100 board lots.
- NT$100 million plus median 20-day volume >= 100 board lots.
- NT$100 million plus median 20-day volume >= 300 board lots.

The board-lot screen prevents a high-priced stock from passing only because a
small number of shares produces a large trading amount.

## Fixed candidate gate

The confirmation period excludes the latest incomplete year. A rule is marked
`strong_candidate` only when it has:

- At least 3 complete confirmation years.
- Positive years in at least 60% of those years and at least 2 years.
- At least 100 events and 30 unique stocks by default.
- Positive 40-day same-day-universe excess return.
- A positive lower bound of the 95% moving-block bootstrap interval.

Passing Phase 6C still does not enable deployment. Phase 6D must train the final
TWSE model, freeze the selected lifecycle rule, and integrate the official TWSE
daily data pipeline and Discord notification flow.

## Run

```powershell
.\scripts\run-institutional-twse-lifecycle.ps1
```

Primary archive:

```text
research/output/twse/phase6c_twse_lifecycle_validation_reports.zip
```
