# `p=1` Solve Speed/Stability Pass (Comprehensive)

*Date: 2026-02-26*

## Scope

Benchmarked `p=1` solve paths with `10` trials and `n=1024`:

- non-SPD suite (`scripts/matrix_solve_nonspd.py`)
  - `k in {1,16,64}`
  - cases: `gaussian_shifted`, `nonnormal_upper`, `similarity_posspec`, `similarity_posspec_hard`
  - added safe-early divergence guard:
    - `--nonspd-safe-fallback-tol 0.01`
    - `--nonspd-safe-early-y-tol 0.8`
- SPD suite (`scripts/matrix_solve.py --p 1`)
  - exact baselines now include `Torch-Solve` and `Torch-Cholesky-Solve`
  - `k in {1,16,64}`
  - core cases: `gaussian_spd`, `illcond_1e6`
  - comprehensive case sweep: +`illcond_1e12`, `near_rank_def`, `spike` (at `k=16`)

Cross-size sanity (`n in {256,512,1024}`) was also run for both suites at `k=16`.

## Artifacts

- non-SPD:
  - `benchmark_results/2026_02_26/nonspd_p1_suite/t10_fp32_safe_early/`
  - `benchmark_results/2026_02_26/nonspd_p1_suite/cross_size_k16_t10_fp32_safe_early/`
  - summary: `benchmark_results/2026_02_26/nonspd_p1_suite/summary_t10_fp32_safe_early.md`
- SPD:
  - `benchmark_results/2026_02_26/spd_p1_suite/t10_fp32_cholesky/`
  - `benchmark_results/2026_02_26/spd_p1_suite/cross_size_k16_t10_fp32_cholesky/`
  - `benchmark_results/2026_02_26/spd_p1_suite/comprehensive_k16_t10_fp32_cholesky/`
  - summaries:
    - `benchmark_results/2026_02_26/spd_p1_suite/summary_t10_fp32_cholesky.md`
    - `benchmark_results/2026_02_26/spd_p1_suite/summary_comprehensive_k16_t10_fp32_cholesky.md`

## Non-SPD Findings (`p=1`)

- Approximate throughput path:
  - `PE-Quad-Coupled-Apply` remains fastest on moderate non-SPD cases.
  - Typical `n=1024` runtime: about `2.6` to `2.9 ms` vs `Torch-Solve` about `4.5` to `4.9 ms`.
- Robust path:
  - `PE-Quad-Coupled-Apply-Safe` with early guard eliminates catastrophic failures on `similarity_posspec_hard`.
  - On hard case, safe mode now lands near solve-level error (same order as `Torch-Solve`) and is materially faster than adaptive mode.
  - Hard-case latency remains higher than direct `Torch-Solve` because exact fallback is intentionally triggered.

## SPD Findings (`p=1`)

- `Torch-Cholesky-Solve` is the strongest exact baseline in this environment:
  - faster than `Torch-Solve` across all tested SPD cases.
- `PE-Quad-Coupled-Apply` is still faster than `Torch-Solve` but slower than `Torch-Cholesky-Solve` at `p=1`.
- No catastrophic stability failures observed for coupled PE across tested SPD cases, but exact baselines keep substantially lower error.

## Practical Policy

- non-SPD, throughput-first:
  - use `PE-Quad-Coupled-Apply`.
- non-SPD, robustness-first:
  - use `PE-Quad-Coupled-Apply-Safe` with safe fallback + early guard.
- SPD `p=1` exact solve baseline:
  - compare against `Torch-Cholesky-Solve` first (not just `Torch-Solve`).
