# Benchmark Decisions

## 2026-02-26: SPD p=1 Torch-Solve backend

Decision:
- Keep `Torch-Solve` on SPD `p=1` locked to Cholesky (`torch.linalg.cholesky` + `torch.cholesky_solve`).
- Keep `Torch-Cholesky-Solve-ReuseFactor` as a separate case when the factorization can be reused across repeated RHS solves.

Benchmark arguments:
- Driver command:
  - `uv run python benchmarks/run_benchmarks.py --trials 10 --only "SPD p=1 k<n" --ab-extra-args-a "--p1-torch-solve-backend linalg" --ab-extra-args-b "--p1-torch-solve-backend cholesky" --ab-label-a linalg --ab-label-b cholesky --ab-out benchmark_results/ab_spd_p1_torchsolve_backend.md --manifest-out benchmark_results/ab_spd_p1_torchsolve_backend_manifest.json`
- Effective benchmark spec:
  - `matrix_solve.py --p 1 --sizes 1024,2048 --k 1,16,64 --trials 10 --timing-reps 5 --timing-warmup-reps 2 --dtype bf16`
  - Cases: `gaussian_spd`, `illcond_1e6` (default in `matrix_solve.py`).
  - A/B-only toggle: `--p1-torch-solve-backend linalg` vs `cholesky`.
- Archived artifacts:
  - `benchmark_results/runs/2026_02_26/ab_spd_p1_torchsolve_backend_173424/`

Key results from the archived report:
- `Torch-Solve` with Cholesky was faster in all 12 SPD `p=1` cells (`n in {1024, 2048}`, `k in {1,16,64}`, cases `gaussian_spd`, `illcond_1e6`).
- Weighted speedup of Cholesky over `linalg.solve`: about `1.99x` lower total ms.
- Geometric mean speedup: about `2.07x`.
- Accuracy also improved (`relerr` lower on Cholesky side in this run).
- For repeated solves with the same matrix, `Torch-Cholesky-Solve-ReuseFactor` was about `1.75x` faster than re-factorizing each call.

## 2026-02-26: SPD p=2/p=4 schedule-trim A/B (`target_err=0.01`)

Decision:
- Keep `--online-coeff-target-interval-err` default at `0.01` for SPD solve benchmarks.

Benchmark arguments (p=2 run):
- Driver command:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n" --ab-extra-args-a "--online-coeff-target-interval-err 0.0" --ab-extra-args-b "--online-coeff-target-interval-err 0.01" --ab-label-a target0 --ab-label-b target001 --ab-out benchmark_results/ab_spd_p2_kltn_default.md --manifest-out benchmark_results/ab_spd_p2_kltn_default_manifest.json`
- Effective benchmark spec:
  - `matrix_solve.py --p 2 --sizes 1024,2048 --k 1,16,64 --trials 5 --timing-reps 5 --timing-warmup-reps 2 --dtype bf16`
  - Cases: `gaussian_spd`, `illcond_1e6`.
- A/B-only toggle: `--online-coeff-target-interval-err 0.0` vs `0.01`.
- Archived artifacts:
  - `benchmark_results/runs/2026_02_26/ab_spd_p2_kltn_default_171528/`

Benchmark arguments (p=4 run):
- Driver command:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n" --ab-extra-args-a "--online-coeff-target-interval-err 0.0" --ab-extra-args-b "--online-coeff-target-interval-err 0.01" --ab-label-a target0 --ab-label-b target001 --ab-out benchmark_results/ab_spd_p4_kltn_default.md --manifest-out benchmark_results/ab_spd_p4_kltn_default_manifest.json`
- Effective benchmark spec:
  - `matrix_solve.py --p 4 --sizes 1024,2048 --k 1,16,64 --trials 5 --timing-reps 5 --timing-warmup-reps 2 --dtype bf16`
  - Cases: `gaussian_spd`, `illcond_1e6`.
- A/B-only toggle: `--online-coeff-target-interval-err 0.0` vs `0.01`.
- Archived artifacts:
  - `benchmark_results/runs/2026_02_26/ab_spd_p4_kltn_default_171756/`

Key results:
- `p=2`: `PE-Quad-Coupled-Apply` improved from `117.617 ms` to `82.896 ms` across 12 cells (`~1.42x` faster, `~29.5%` lower total ms), with relerr ratio `B/A ~= 1.029`.
- `p=4`: `PE-Quad-Coupled-Apply` improved from `135.321 ms` to `106.090 ms` across 12 cells (`~1.28x` faster, `~21.6%` lower total ms), with relerr ratio `B/A ~= 0.959`.
