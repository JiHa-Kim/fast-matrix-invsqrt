# Solver Performance (Fresh Run)

Date: 2026-02-26

## Source Logs
- `benchmark_results/spd_p1_n1024_k1-1024_20260226_155655.log`
- `benchmark_results/spd_p2_n1024_k1-1024_20260226_155655.log`
- `benchmark_results/spd_p4_n1024_k1-1024_20260226_155655.log`
- `benchmark_results/nonspd_n1024_k1-1024_20260226_155655.log`
- `benchmark_results/nonspd_n2048_k1-2048_20260226_155655.log` (empty)

## Notes
- Batch benchmark runner executed with `benchmarks/run_benchmarks.py`.
- Newton baselines were included in direct solve suites:
  - `Inverse-Newton-Inverse-Multiply`
  - `Inverse-Newton-Coupled-Apply`
- Non-SPD `n=2048` output file is empty in this run and should be rerun if needed.

## High-Level Outcome
- SPD p=1: Newton-coupled and Cholesky-solve are both strong low-latency baselines depending on case and k.
- Non-SPD p=1: Newton-coupled was consistently fastest among finite runs but with large relative errors on hard cases.
