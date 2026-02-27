# Benchmark Decisions

## 2026-02-26: Kappa-bin minimax lookup candidate (k<n-only wiring)

Decision:
- Reject `kappa-minimax-table-klt-only`.
- Keep `greedy-affine-opt` as default for coupled PE scheduling.

Benchmark arguments:
- `p=2`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode kappa-minimax-table-klt-only" --ab-label-a greedy_affine_opt --ab-label-b kappa_table_klt_only --ab-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p2/manifest.json`
- `p=4`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode kappa-minimax-table-klt-only" --ab-label-a greedy_affine_opt --ab-label-b kappa_table_klt_only --ab-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_kappa_table_klt_only_p4/manifest.json`

Key results (`PE-Quad-Coupled-Apply`):
- `p=2`: slight gain overall (`-1.20%`), with `k<n` gain (`-4.75%`) but `k=n` regression (`+2.55%`), relerr stable.
- `p=4`: clear regression (`+17.29%` total), including `k<n` (`+28.74%`) and `k=n` (`+7.80%`).
- Relerr drift was not large in this run, but speed regressions were decisive.

## 2026-02-26: Staged minimax+Newton-tail schedule candidate (`p=2,4`)

Decision:
- Reject staged candidate (`greedy-minimax` early + forced inverse-Newton tail).
- Keep `greedy-affine-opt` as default coupled schedule mode.

Benchmark arguments:
- `p=2`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode staged-minimax-newton --online-coeff-staged-newton-tail 2" --ab-label-a greedy_affine_opt --ab-label-b staged_minimax_newton --ab-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p2/manifest.json`
- `p=4`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024" --ab-extra-args-a="--online-coeff-mode greedy-affine-opt" --ab-extra-args-b="--online-coeff-mode staged-minimax-newton --online-coeff-staged-newton-tail 2" --ab-label-a greedy_affine_opt --ab-label-b staged_minimax_newton --ab-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_staged_minimax_newton_p4/manifest.json`

Key results (`PE-Quad-Coupled-Apply`):
- `p=2`: improved (`-12.04%`) but with noticeable relerr drift on some cells.
- `p=4`: regressed (`+7.66%`), with both `k<n` and `k=n` slower.

## 2026-02-26: Chebyshev CUDA graph on default path (`p=2,4`)

Decision:
- Enable `Chebyshev-Apply` CUDA graph replay via `--cheb-cuda-graph` independently from global `--cuda-graph`.
- Keep `--cheb-cuda-graph` default enabled.

Benchmark arguments:
- `p=2`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --ab-extra-args-a=--no-cheb-cuda-graph --ab-extra-args-b=--cheb-cuda-graph --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_graph_defaultpath_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_graph_defaultpath_p2/manifest.json`
- `p=4`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024,SPD p=4 k=n=2048" --ab-extra-args-a=--no-cheb-cuda-graph --ab-extra-args-b=--cheb-cuda-graph --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_graph_defaultpath_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_graph_defaultpath_p4/manifest.json`

Key results (`Chebyshev-Apply`):
- `p=2`: aggregate total ms `-12.98%`, `k<n` `-36.72%`, `k=n` `-2.96%`, relerr unchanged.
- `p=4`: aggregate total ms `-16.15%`, `k<n` `-45.13%`, `k=n` `-2.72%`, relerr unchanged.
- Non-Chebyshev methods were effectively unchanged on aggregate.

## 2026-02-26: Global CUDA graph default (`off` vs `on`) for `p=2,4`

Decision:
- Keep global `--cuda-graph` as opt-in (default off).
- Do not switch default-on across all methods.

Benchmark arguments:
- `p=2`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --ab-extra-args-b=--cuda-graph --ab-label-a cuda_graph_off --ab-label-b cuda_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p2/manifest.json`
- `p=4`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024,SPD p=4 k=n=2048" --ab-extra-args-b=--cuda-graph --ab-label-a cuda_graph_off --ab-label-b cuda_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p4/manifest.json`

Key results:
- `p=2` aggregate: `-5.08%`, with wins in both `k<n` and `k=n`.
- `p=4` aggregate: `-1.12%`, but `k=n` regressed (`+4.09%`).
- Method-level mixed behavior outside Chebyshev; relerr unchanged.

## 2026-02-26: Chebyshev mode `fixed` vs `minimax-auto` for `p=2,4`

Decision:
- Keep `--cheb-mode fixed` as the maintained default for `p=2,4` benchmark runs.
- Reject `minimax-auto` as a default policy under the tested matrix (`k<n` and `k=n`).

Benchmark arguments:
- `p=2`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --ab-extra-args-a="--cheb-mode fixed" --ab-extra-args-b="--cheb-mode minimax-auto" --ab-label-a cheb_fixed --ab-label-b cheb_minimax_auto --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p2/manifest.json`
- `p=4`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024,SPD p=4 k=n=2048" --ab-extra-args-a="--cheb-mode fixed" --ab-extra-args-b="--cheb-mode minimax-auto" --ab-label-a cheb_fixed --ab-label-b cheb_minimax_auto --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p4/manifest.json`

Key results (`Chebyshev-Apply`):
- `p=2`: near-neutral aggregate (`-0.08%`), but `k<n` regressed (`+1.57%`); relerr unchanged.
- `p=4`: aggregate regression (`+2.05%`), with regressions for both `k<n` (`+5.66%`) and `k=n` (`+0.57%`); relerr unchanged.

## 2026-02-26: Chebyshev CUDA graph replay for `p=2,4`

Decision:
- Enable CUDA graph replay for `Chebyshev-Apply` when `--cuda-graph` is enabled (default on via `--cheb-cuda-graph`).

Benchmark arguments:
- `p=2`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --extra-args=--cuda-graph --ab-extra-args-a=--no-cheb-cuda-graph --ab-extra-args-b=--cheb-cuda-graph --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p2/manifest.json`
- `p=4`:
  - `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024,SPD p=4 k=n=2048" --extra-args=--cuda-graph --ab-extra-args-a=--no-cheb-cuda-graph --ab-extra-args-b=--cheb-cuda-graph --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p4/manifest.json`
- Note: this A/B was split by `p` to avoid key collisions in combined reports.

Key results (`Chebyshev-Apply`, `k<n` and `k=n` included):
- `p=2`: `19/20` cells faster, `-13.39%` aggregate total ms, relerr unchanged.
  - `k<n`: `-31.16%`; `k=n`: `-6.37%`.
- `p=4`: `17/20` cells faster, `-16.35%` aggregate total ms, relerr unchanged.
  - `k<n`: `-40.22%`; `k=n`: `-6.36%`.

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

## 2026-02-27: non-SPD `p=1` smooth damping prototype (branch-light stability)

Decision:
- Reject as default (and revert prototype code).
- Keep current non-SPD safety policy (`adaptive` / safe fallback guards) for now.

Why tested:
- We explored a branch-light alternative to adaptive branch switching:
  per-step smooth damping of PE coefficients based on cheap `Y` proxy metrics.

Benchmark commands (focused):
- Baseline:
  - `uv run python -m benchmarks.solve.matrix_solve_nonspd --sizes 1024 --k 64 --cases gaussian_shifted,nonnormal_upper,similarity_posspec --methods PE-Quad-Coupled-Apply --trials 3 --timing-reps 3 --timing-warmup-reps 1 --dtype bf16`
- Smooth damping (aggressive):
  - `uv run python -m benchmarks.solve.matrix_solve_nonspd --sizes 1024 --k 64 --cases gaussian_shifted,nonnormal_upper,similarity_posspec --methods PE-Quad-Coupled-Apply --trials 3 --timing-reps 3 --timing-warmup-reps 1 --dtype bf16 --nonspd-smooth-damping --nonspd-smooth-damping-gain 1.0 --nonspd-smooth-damping-min 0.3 --nonspd-smooth-damping-metric diag`
- Smooth damping (mild):
  - `uv run python -m benchmarks.solve.matrix_solve_nonspd --sizes 1024 --k 64 --cases gaussian_shifted,nonnormal_upper,similarity_posspec --methods PE-Quad-Coupled-Apply --trials 3 --timing-reps 3 --timing-warmup-reps 1 --dtype bf16 --nonspd-smooth-damping --nonspd-smooth-damping-gain 0.1 --nonspd-smooth-damping-min 0.8 --nonspd-smooth-damping-metric diag`
- Hard case check:
  - `uv run python -m benchmarks.solve.matrix_solve_nonspd --sizes 1024 --k 64 --cases similarity_posspec_hard --methods PE-Quad-Coupled-Apply --trials 3 --timing-reps 3 --timing-warmup-reps 1 --dtype bf16`
  - `uv run python -m benchmarks.solve.matrix_solve_nonspd --sizes 1024 --k 64 --cases similarity_posspec_hard --methods PE-Quad-Coupled-Apply --trials 3 --timing-reps 3 --timing-warmup-reps 1 --dtype bf16 --nonspd-smooth-damping --nonspd-smooth-damping-gain 0.1 --nonspd-smooth-damping-min 0.8 --nonspd-smooth-damping-metric diag`

Key results:
- Aggressive damping (`gain=1.0`, `min=0.3`) regressed both speed and relerr on normal non-SPD cases.
- Mild damping (`gain=0.1`, `min=0.8`) was mostly neutral/slightly mixed on normal cases.
- Hard case `similarity_posspec_hard` remained at `relerr = 1.000e+00` with and without damping (no fundamental stability win).
- Conclusion: this mechanism is not strong enough to replace current safety branching and not worth keeping as a default path.
