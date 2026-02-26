# Solve p=1 Comprehensive (Fresh)

Date: 2026-02-26

## SPD (n=1024)
From `benchmark_results/spd_p1_n1024_k1-1024_20260226_155655.log`:
- Added Newton baselines are present in all cells.
- Typical pattern in this run:
  - `Inverse-Newton-Coupled-Apply` is the fastest iterative direct-apply baseline.
  - `Torch-Cholesky-Solve` remains highly competitive with near-zero relerr.
  - PE-Quad coupled/apply stays close while retaining schedule flexibility.

## non-SPD (n=1024)
From `benchmark_results/nonspd_n1024_k1-1024_20260226_155655.log`:
- `BEST finite` repeatedly selected `Inverse-Newton-Coupled-Apply` for speed.
- Relative error is high on hard non-normal cases (up to ~1.0), so accuracy-safe modes remain important.

## Gaps
- `benchmark_results/nonspd_n2048_k1-2048_20260226_155655.log` is empty and should be rerun for full coverage.
