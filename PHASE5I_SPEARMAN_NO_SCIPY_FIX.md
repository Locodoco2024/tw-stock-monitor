# Phase 5I Spearman no-SciPy fix

- Replaced `Series.corr(method="spearman")`, which imports SciPy at runtime.
- Spearman is now calculated as Pearson correlation of average ranks using pandas only.
- No new dependency is required.
- Added regression tests that explicitly reject any SciPy import.
