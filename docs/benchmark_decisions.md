# Benchmark Decisions

## 2026-02-27: non-SPD `p=1` monotone residual damping safeguard

Decision:
- Reject and archive this prototype.
- Revert the code path/flags from active sources; keep benchmark artifacts for record.

Why tested:
- We tested an opt-in monotone residual safeguard for non-SPD `p=1` coupled apply:
  per checked step, retry with damped coefficients (`omega = 1, 1/2, 1/4`) and accept only residual-nondegrading updates.

Benchmark arguments:
- Focused A/B on maintained non-SPD `p=1 k<n` slice:
  - `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --ab-extra-args-a="--no-nonspd-monotone-residual" --ab-extra-args-b="--nonspd-monotone-residual --nonspd-monotone-growth-tol 1.0 --nonspd-monotone-check-every 1 --nonspd-monotone-backtracks 2" --ab-label-a baseline --ab-label-b monotone --ab-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_monotone_step1/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_monotone_step1/manifest.json`

Key results:
- Accuracy was unchanged in matched rows (`relerr_ratio(B/A)=1.000`).
- Runtime regressed materially across all measured cells (roughly `+9%` to `+156%` total ms).
- Conclusion: no robustness gain in this matrix with clear cost regression, so it is archived and not kept in active code.

## 2026-02-27: non-SPD `p=1` affine-only schedule policy

Decision:
- Reject and archive as a default policy.
- Revert the experimental schedule-planner/CLI wiring from active code.

Why tested:
- We evaluated an affine-only schedule policy for non-SPD `p=1` coupled apply:
  disable quadratic PE steps and select only among affine candidates
  (inverse-Newton and interval-optimal affine).

Benchmark arguments:
- Focused A/B on maintained non-SPD `p=1 k<n` slice:
  - `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --ab-extra-args-a="--no-p1-affine-only-schedule" --ab-extra-args-b="--p1-affine-only-schedule" --ab-label-a baseline --ab-label-b affine_only --ab-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_affine_only_step2/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_affine_only_step2/manifest.json`

Key results:
- Not a strict win: behavior was highly mixed.
- Large regressions in many cells (often around `+250%` total ms), with a few speed wins on specific cases.
- Accuracy was also mixed (some cells improved, some worsened).
- Conclusion: unsuitable as a maintained default; archived for record only.

## 2026-02-27: Dual Gram-RHS apply path (`apply_inverse_root_gram_rhs_spd`)

Decision:
- Keep and expose the dual path for workloads with RHS in `range(G^T)` (`M = G^T B`).
- Include Gram-RHS comparison cases in the maintained benchmark driver (`run_benchmarks.py`) so the path is continuously measurable with standard reports/manifests.

Benchmark arguments:
- Standard-suite filtered run:
  - `uv run python benchmarks/run_benchmarks.py --only "GRAM RHS" --markdown --out benchmark_results/runs/2026_02_27/gram_rhs_standard_suite/report.md --manifest-out benchmark_results/runs/2026_02_27/gram_rhs_standard_suite/manifest.json`

Key results (`m=256`, `n=1024`, `k in {1,16,64}`, `dtype=bf16`, `precond=jacobi`):
- `p=2`:
  - `k=1`: dual `1.223 ms` vs primal `2.385 ms` (`1.95x` faster)
  - `k=16`: dual `1.248 ms` vs primal `2.376 ms` (`1.90x` faster)
  - `k=64`: dual `1.301 ms` vs primal `1.897 ms` (`1.46x` faster)
- `p=4`:
  - `k=1`: dual `2.241 ms` vs primal `3.002 ms` (`1.34x` faster)
  - `k=16`: dual `1.859 ms` vs primal `2.370 ms` (`1.28x` faster)
  - `k=64`: dual `1.209 ms` vs primal `2.343 ms` (`1.94x` faster)
- Accuracy parity in-run: primal and dual rows reported identical `relerr` per cell.

## 2026-02-27: Gram precondition cache reuse (`reuse_precond=True`) validation sweep

Decision:
- Keep `apply_inverse_root_gram_spd(..., reuse_precond=True)` as the recommended path when the same Gram source `G` is reused across solves.
- Standard full-suite A/B (non-Gram-focused) was near-neutral overall, while Gram-focused sweep showed consistent large wins with exact output parity.

Benchmark arguments:
- Full maintained matrix (baseline-row A/B):
  - `uv run python benchmarks/run_benchmarks.py --markdown --ab-baseline-rows-in baseline_rows.json --ab-label-a baseline --ab-label-b gram_cached --ab-out benchmark_results/runs/2026_02_27/ab_fullsuite_gram_cached/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_fullsuite_gram_cached/manifest.json`
- Gram-focused sweep (directly exercises new path):
  - `uv run python -m benchmarks.solve.matrix_solve_gram --p 2 --m 2048 --n 512 --k 16 --trials 10 --timing-reps 3 --warmup-reps 1 --dtype fp32 --gram-mode col-norm --precond-mode jacobi --markdown --out benchmark_results/runs/2026_02_27/gram_reuse_sweep/p2_m2048_n512_k16.md`
  - `uv run python -m benchmarks.solve.matrix_solve_gram --p 2 --m 2048 --n 512 --k 64 --trials 10 --timing-reps 3 --warmup-reps 1 --dtype fp32 --gram-mode col-norm --precond-mode jacobi --markdown --out benchmark_results/runs/2026_02_27/gram_reuse_sweep/p2_m2048_n512_k64.md`
  - `uv run python -m benchmarks.solve.matrix_solve_gram --p 4 --m 2048 --n 512 --k 16 --trials 10 --timing-reps 3 --warmup-reps 1 --dtype fp32 --gram-mode col-norm --precond-mode jacobi --markdown --out benchmark_results/runs/2026_02_27/gram_reuse_sweep/p4_m2048_n512_k16.md`
  - `uv run python -m benchmarks.solve.matrix_solve_gram --p 4 --m 2048 --n 512 --k 64 --trials 10 --timing-reps 3 --warmup-reps 1 --dtype fp32 --gram-mode col-norm --precond-mode jacobi --markdown --out benchmark_results/runs/2026_02_27/gram_reuse_sweep/p4_m2048_n512_k64.md`
  - `uv run python -m benchmarks.solve.matrix_solve_gram --p 2 --m 4096 --n 1024 --k 64 --trials 10 --timing-reps 3 --warmup-reps 1 --dtype fp32 --gram-mode col-norm --precond-mode jacobi --markdown --out benchmark_results/runs/2026_02_27/gram_reuse_sweep/p2_m4096_n1024_k64.md`
  - `uv run python -m benchmarks.solve.matrix_solve_gram --p 4 --m 4096 --n 1024 --k 64 --trials 10 --timing-reps 3 --warmup-reps 1 --dtype fp32 --gram-mode col-norm --precond-mode jacobi --markdown --out benchmark_results/runs/2026_02_27/gram_reuse_sweep/p4_m4096_n1024_k64.md`

Key results:
- Full maintained matrix A/B (`ab_fullsuite_gram_cached`):
  - Matched aggregate delta (B vs A): `+0.62%` total ms (near-neutral, mixed by cell).
  - Relative error unchanged on matched rows (`relerr_ratio(B/A)=1.000`).
- Gram-focused sweep (`reuse_precond=True` vs `False`):
  - `p=2, 2048x512, k=16`: `2.823x` faster
  - `p=2, 2048x512, k=64`: `2.936x` faster
  - `p=2, 4096x1024, k=64`: `1.883x` faster
  - `p=4, 2048x512, k=16`: `2.634x` faster
  - `p=4, 2048x512, k=64`: `2.587x` faster
  - `p=4, 4096x1024, k=64`: `1.728x` faster
  - Geometric-mean speedup across sweep: `2.384x`
  - Output parity: all sweep cells reported relative diff `0.000e+00`

## 2026-02-27: Gram precondition cache reuse (`reuse_precond=True`)

Decision:
- Keep `apply_inverse_root_gram_spd(..., reuse_precond=True)` as the preferred path for repeated solves with fixed `G`.
- The cache-reuse path is materially faster with identical outputs in the measured run.

Benchmark arguments:
- `uv run python -m benchmarks.solve.matrix_solve_gram --p 2 --m 2048 --n 512 --k 64 --trials 12 --timing-reps 3 --warmup-reps 1 --dtype fp32 --gram-mode col-norm --precond-mode jacobi --markdown --out benchmark_results/runs/2026_02_27/gram_reuse_precond_p2_cpu/report.md`

Key results:
- `reuse_precond=False`: `3.543 ms`
- `reuse_precond=True`: `1.202 ms`
- Speedup: `2.947x`
- Output parity: relative diff `0.000e+00`

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

## 2026-02-27: fused-addmm implementation optimizations (Clenshaw and RHS-terminal)

Decision:
- Reject `fused-addmm` optimizations for default path.
- Keep existing `_matmul_into` + elementwise `add_`/`mul_` sequence.

Why tested:
- Attempted to reduce kernel launch overhead by fusing matmuls with linear combinations using `torch.addmm` in `apply_inverse_chebyshev_with_coeffs` and `_apply_quadratic_left_rhs_terminal`.

Benchmark arguments:
- Baseline: `155131_fused_addmm_baseline`
- A/B Test: `155300_fused_addmm_ab_test`
- command: `uv run python -m benchmarks.run_benchmarks --markdown --ab-baseline-rows-in baseline_rows.json --trials 5 --timing-reps 5 --only "p=2,p=4"`

Key results:
- `p=2` aggregate speed: `-8.32%` total ms (win).
- `p=4` aggregate speed: `+1.35%` total ms (regression/neutral).
- Accuracy regression: Consistent `relerr_ratio (B/A)` of `1.18x` to `1.33x` across most `p=2,4` cells in `bf16`.
- Conclusion: While `addmm` reduces kernel launches, it appears to introduce accumulation noise in `bf16` that regresses relative error by ~20-30%. Since there is no "essentially strict win" on both speed and accuracy (especially for `p=4`), the change is rejected.

## 2026-02-27: Block-Jacobi preconditioning for p=1

Decision:
- Reject `block-jacobi` as default preconditioner.
- Keep `jacobi` (diagonal) or `row-norm` as defaults.
- Keep the `block-jacobi` implementation as an optional mode (`--precond block-jacobi`).

Why tested:
- Block-diagonal scaling can theoretically cluster eigenvalues better than simple diagonal scaling for matrices with block structure.

Benchmark arguments:
- Run: `2026_02_27/ab_precond_block_jacobi_p1_v3`
- command: `uv run python -m benchmarks.run_benchmarks --only "SPD p=1" --ab-extra-args-a="--precond jacobi" --ab-extra-args-b="--precond block-jacobi"`

Key results:
- Speed regression: `block-jacobi` pre-processing (`ms_precond`) is significantly more expensive than `jacobi` (e.g., ~2.5ms vs ~1.2ms for 1024x1024).
- Iteration win: It does reduce iteration time in some cases (e.g., ~0.8ms vs ~0.85ms), but not enough to offset the pre-processing cost for the tested RHS counts (k=1 to 1024).
- Total time regression: Aggregate total ms increased (regressed) by ~20-50% across most cells.
- Conclusion: The overhead of batched EVD/inversion for 32x32 blocks is too high for a single-solve preconditioning step at these matrix sizes. It remains available as an opt-in for specialized workloads where the preconditioner can be heavily reused.

## 2026-02-27: lambda_min power iteration estimation

Decision:
- Reject power iteration approach for `lambda_min` estimation.

Why tested:
- Attempted to calculate a tight `l_min` bound for PE/Chebyshev polynomials dynamically using shifted power iterations to improve initial convergence.

Benchmark arguments:
- Run: `2026_02_27/ab_precond_power_lmin_p2_v2`
- command: `uv run python -m benchmarks.run_benchmarks --only "SPD p=2" --ab-extra-args-a="--lambda-max-est row_sum --l-target 0.05" --ab-extra-args-b="--lambda-max-est row_sum --lambda-min-est power --l-target 0.05"`

Key results:
- Speed regression: The power iteration overhead was significant, causing total times to regress heavily (e.g. +35% to +68% on smaller sizes) without sufficient reduction in the number of PE steps.
- Conclusion: The overhead of estimating `lambda_min` via shifted power iteration outweighs the benefits of a tighter interval for these matrix sizes. It was rejected and reverted.

## 2026-02-27: Matrix-Free Chebyshev Apply for Gram Matrices

Decision:
- Keep the `apply_inverse_proot_chebyshev_gram` implementation as the preferred path for $b \ll n$ workloads.
- Avoid forming the dense $n \times n$ matrix $A = G^T G$.

Why tested:
- For ML workloads where features are represented by a wide matrix $G$ ($b \times n$ where $b \ll n$), forming the dense Gram matrix $G^T G$ takes $O(n^2 b)$ compute and $O(n^2)$ memory.
- We implemented a Matrix-Free path that computes $A(Y) = G^T (G Y)$, passing this callable to the Chebyshev Clenshaw recurrence.

Benchmark arguments:
- Run: Standalone `bench_gram.py`
- $n = 4096$, $b \in \{64, 256\}$, $k = 64$.
- Compared `apply_inverse_sqrt_gram_spd` (Dense PE-Quad) vs `apply_inverse_proot_chebyshev_gram` (Matrix-Free Chebyshev, degree 32).

Key results:
- Speedup: Matrix-Free was consistently $>10\times$ faster.
  - $b=64$: Matrix-Free `14.11 ms` vs Dense `162.15 ms`.
  - $b=256$: Matrix-Free `11.25 ms` vs Dense `163.34 ms`.
- Memory: By avoiding the dense $4096 \times 4096$ allocation, memory overhead was dramatically reduced.
- Conclusion: The matrix-free Chebyshev path is a strict and massive win for wide Gram matrices. The changes have been kept and integrated.
