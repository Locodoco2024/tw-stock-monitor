# Phase 5H partial latest-date seed fix

- Seed dates are selected only when TPEx active/training universe coverage reaches 70% (minimum 50 stocks).
- A partially downloaded newest date is skipped instead of being treated as the seed endpoint.
- Coverage requirements remain unchanged; the fix does not lower the 70% threshold.
- If fewer than 60 complete market dates exist, the error now includes the latest partial date and stock count.
