# Phase 5I outcome column compatibility fix

- Phase 3 shards contain 5/10/20-day extrema only.
- 40-day extrema are loaded from `research/data/phase4d_cache/*/<stock_id>_40d.csv.gz` when available.
- Missing Phase 4D cache no longer aborts Phase 5I; 40-day extrema remain blank while 20-day entry-risk research continues.
- Added audit counts for available 20-day and 40-day extrema.
