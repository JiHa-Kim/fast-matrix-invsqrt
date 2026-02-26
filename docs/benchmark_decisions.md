# Benchmark Decisions

## 2026-02-26: SPD p=1 Torch-Solve backend

Decision:
- Keep `Torch-Solve` on SPD `p=1` locked to Cholesky (`torch.linalg.cholesky` + `torch.cholesky_solve`).
- Keep `Torch-Cholesky-Solve-ReuseFactor` as a separate case when the factorization can be reused across repeated RHS solves.

Evidence:
- A/B run: linalg vs cholesky backend for `Torch-Solve`.
- Archived artifacts: `benchmark_results/runs/2026_02_26/ab_spd_p1_torchsolve_backend_173424/`.
- Command used:
  - `uv run python benchmarks/run_benchmarks.py --trials 10 --only "latest_spd_solve_logs\\spd_p1_klt_n_" --ab-extra-args-a "--p1-torch-solve-backend linalg" --ab-extra-args-b "--p1-torch-solve-backend cholesky" --ab-label-a linalg --ab-label-b cholesky --ab-out benchmark_results/ab_spd_p1_torchsolve_backend.md --manifest-out benchmark_results/ab_spd_p1_torchsolve_backend_manifest.json`

Key results from the archived report:
- `Torch-Solve` with Cholesky was faster in all 12 SPD `p=1` cells (`n in {1024, 2048}`, `k in {1,16,64}`, cases `gaussian_spd`, `illcond_1e6`).
- Weighted speedup of Cholesky over `linalg.solve`: about `1.99x` lower total ms.
- Geometric mean speedup: about `2.07x`.
- Accuracy also improved (`relerr` lower on Cholesky side in this run).
- For repeated solves with the same matrix, `Torch-Cholesky-Solve-ReuseFactor` was about `1.75x` faster than re-factorizing each call.
