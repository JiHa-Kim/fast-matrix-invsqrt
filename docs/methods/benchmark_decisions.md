# Benchmark Decisions

This document tracks architectural and policy decisions made based on empirical benchmark results. It serves as an Architecture Decision Record (ADR) for the performance-critical paths of the project.

## Table of Contents

- [2026-02-28: Benchmark assessment overhaul (quality + stability + efficiency)](#2026-02-28-benchmark-assessment-overhaul-quality--stability--efficiency)
- [2026-02-27: non-SPD `p=1` coupled renormalization policy (`renorm_every`)](#2026-02-27-non-spd-p1-coupled-renormalization-policy-renorm_every)
- [2026-02-27: non-SPD `p=1` monotone residual damping safeguard](#2026-02-27-non-spd-p1-monotone-residual-damping-safeguard)
- [2026-02-27: non-SPD `p=1` affine-only schedule policy](#2026-02-27-non-spd-p1-affine-only-schedule-policy)
- [2026-02-27: non-SPD `p=1` freeze-then-refine (PE -> NSRC) policy](#2026-02-27-non-spd-p1-freeze-then-refine-pe---nsrc-policy)
- [2026-02-27: non-SPD `p=1` preconditioner policy (row-norm vs ruiz)](#2026-02-27-non-spd-p1-preconditioner-policy-row-norm-vs-ruiz)
- [2026-02-27: non-SPD `p=1` main-path adaptive toggle](#2026-02-27-non-spd-p1-main-path-adaptive-toggle)
- [2026-02-27: SPD `p=1` Chebyshev method exposure in benchmark suite](#2026-02-27-spd-p1-chebyshev-method-exposure-in-benchmark-suite)
- [2026-02-27: SPD `p=1` Chebyshev mode (fixed vs minimax-auto)](#2026-02-27-spd-p1-chebyshev-mode-fixed-vs-minimax-auto)
- [2026-02-27: SPD `p=1` Chebyshev k<n degree cap (24 vs 16)](#2026-02-27-spd-p1-chebyshev-kltn-degree-cap-24-vs-16)
- [2026-02-27: SPD `p=1` Chebyshev CUDA graph replay (off vs on)](#2026-02-27-spd-p1-chebyshev-cuda-graph-replay-off-vs-on)
- [2026-02-27: SPD `p=1` policy compare (PE-Quad vs Chebyshev+graph)](#2026-02-27-spd-p1-policy-compare-pe-quad-vs-chebyshevgraph)
- [2026-02-27: non-SPD `p=1` safety guards (on vs off)](#2026-02-27-non-spd-p1-safety-guards-on-vs-off)
- [2026-02-27: non-SPD `p=1` final safety check diagonal gate](#2026-02-27-non-spd-p1-final-safety-check-diagonal-gate)
- [2026-02-27: Dual Gram-RHS apply path](#2026-02-27-dual-gram-rhs-apply-path-apply_inverse_root_gram_rhs_spd)
- [2026-02-27: Gram precondition cache reuse validation sweep](#2026-02-27-gram-precondition-cache-reuse-reuse_precondtrue-validation-sweep)
- [2026-02-26: Kappa-bin minimax lookup candidate](#2026-02-26-kappa-bin-minimax-lookup-candidate-klt-only-wiring)
- [2026-02-26: Staged minimax+Newton-tail schedule candidate](#2026-02-26-staged-minimaxnewton-tail-schedule-candidate-p24)
- [2026-02-26: Chebyshev CUDA graph on default path](#2026-02-26-chebyshev-cuda-graph-on-default-path-p24)
- [2026-02-26: Global CUDA graph default](#2026-02-26-global-cuda-graph-default-off-vs-on-for-p24)
- [2026-02-26: Chebyshev mode fixed vs minimax-auto for p=2,4](#2026-02-26-chebyshev-mode-fixed-vs-minimax-auto-for-p24)
- [2026-02-26: SPD p=1 Torch-Solve backend](#2026-02-26-spd-p1-torch-solve-backend)
- [2026-02-26: SPD p=2/p=4 schedule-trim A/B (target_err=0.01)](#2026-02-26-spd-p2p4-schedule-trim-ab-target_err001)
- [2026-02-27: non-SPD `p=1` smooth damping prototype](#2026-02-27-non-spd-p1-smooth-damping-prototype-branch-light-stability)
- [2026-02-27: fused-addmm implementation optimizations](#2026-02-27-fused-addmm-implementation-optimizations-clenshaw-and-rhs-terminal)
- [2026-02-27: Block-Jacobi preconditioning for p=1](#2026-02-27-block-jacobi-preconditioning-for-p1)
- [2026-02-27: lambda_min power iteration estimation](#2026-02-27-lambda_min-power-iteration-estimation)
- [2026-02-27: Matrix-Free Chebyshev Apply for Gram Matrices](#2026-02-27-matrix-free-chebyshev-apply-for-gram-matrices)

---

## 2026-02-28: Benchmark assessment overhaul (quality + stability + efficiency)

Decision:
- Keep and adopt the new benchmark assessment schema and reporting policy.
- Continue using compatibility parsing for historical logs/rows.

What changed:
- Solver rows now include:
  - `relerr` (median relative error),
  - `relerr_p90` (tail error),
  - `fail_rate` (non-finite output rate),
  - `q_per_ms` (`max(0, -log10(relerr)) / iter_ms`).
- A/B markdown now reports deltas/ratios for these quality and stability fields, not only runtime and median error.
- Standard markdown now includes an `Assessment Leaders` table per scenario (`kind,p,n,k,case`) with score:
  - score = `q_per_ms / max(1, relerr_p90/relerr) * (1 - fail_rate)`.
- Parser now preserves rows where values are `inf`/`nan` (instead of dropping failed cells).

Why:
- Prior benchmark policy over-indexed on `total_ms` and median `relerr`, which can hide tail risk and silent instability in non-SPD or low-precision regimes.
- The new schema makes benchmark-driven decisions more robust by explicitly incorporating failure behavior and tail quality.

Compatibility:
- Older logs without new fields are still parsed (missing metrics default to `NaN`).
- Existing rows caches remain readable; new caches are written as `solver_benchmark_rows.v2`.

---

## 2026-02-27: non-SPD `p=1` coupled renormalization policy (`renorm_every`)

Decision:
- Reject `renorm_every=1` as default policy for maintained non-SPD `p=1` benchmark runs.
- Keep renormalization support in code as an optional tuning knob, but default to `renorm_every=0`.

Why tested:
- We added coupled-state renormalization (`Y` recentering with consistent `Z/X` scaling) to mitigate drift.
- This run evaluates whether periodic renorm gives a strict win in maintained non-SPD `p=1` cells.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --trials 5 --timing-reps 5 --timing-warmup-reps 2 --ab-extra-args-a="--methods PE-Quad-Coupled-Apply --renorm-every 0" --ab-extra-args-b="--methods PE-Quad-Coupled-Apply --renorm-every 1" --ab-label-a renorm_off --ab-label-b renorm_on --run-name ab_nonspd_p1_renorm_step13 --ab-match-on-method --ab-interleave --integrity-checksums`

Key results (`benchmark_results/runs/2026_02_27/225808_ab_nonspd_p1_renorm_step13/solver_benchmarks_ab.md`):
- Speed: `renorm_on` was slower in `12/12` matched cells (`B faster = 0/12`).
- Quality: mixed; `renorm_on` was better-or-equal on quality metrics in only `6/12` cells.
- Composite assessment: `renorm_on` improved score in only `1/12` cells.
- Failures: no fail-rate improvement (`0.0%` both sides in all cells).

Conclusion:
- This is not a strict win and is rejected as default policy.
- Default benchmark setting is reverted to `--renorm-every 0`; optional knob remains for future targeted experiments.

---

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

## 2026-02-27: non-SPD `p=1` freeze-then-refine (`PE -> NSRC`) policy

Decision:
- Reject and archive as default behavior.
- Revert benchmark-gated policy wiring from active code.

Why tested:
- We evaluated a two-phase non-SPD `p=1` policy:
  build a frozen preconditioner with a few PE steps, then run additive NSRC refinement
  (with the same final residual fallback tolerance used by baseline).

Benchmark arguments:
- Focused A/B on maintained non-SPD `p=1 k<n` slice:
  - `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --ab-extra-args-a="--no-p1-freeze-refine" --ab-extra-args-b="--p1-freeze-refine --p1-freeze-pe-steps 2 --p1-freeze-ref-steps 3" --ab-label-a baseline --ab-label-b freeze_refine --ab-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_freeze_refine_step3/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_freeze_refine_step3/manifest.json`

Key results:
- Not a strict win: speed and accuracy were both mixed by case.
- Several cells regressed in total ms (notably `nonnormal_upper` and `*_hard` cases),
  while some cells were neutral/slightly faster.
- Accuracy improved on `gaussian_shifted` and `nonnormal_upper` cells but worsened on
  `similarity_posspec`; hard-case accuracy stayed unchanged due fallback.
- Conclusion: this configuration is not robustly better in the maintained matrix and is archived.

## 2026-02-27: non-SPD `p=1` preconditioner policy (`row-norm` vs `ruiz`)

Decision:
- Keep `row-norm` as maintained default for the non-SPD `p=1` benchmark matrix.
- Reject switching default to `ruiz` (`ruiz_iters=2`) based on this focused A/B.

Why tested:
- The p=1 notes suggest stronger balancing may help non-normal stability.
- We ran a direct one-change A/B on maintained `k<n` cells.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --ab-extra-args-a="--precond row-norm --precond-ruiz-iters 2" --ab-extra-args-b="--precond ruiz --precond-ruiz-iters 2" --ab-label-a row_norm --ab-label-b ruiz --ab-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_precond_ruiz_step4/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_precond_ruiz_step4/manifest.json`

Key results:
- Mixed behavior with broad total-time regressions from higher preconditioning cost.
- `ruiz` reduced iteration time in some cells, but preconditioning overhead dominated.
- Accuracy changes were mixed (some cells better, some worse), with no decisive global gain.
- Conclusion: no strict win; keep `row-norm` as default and archive this run.

## 2026-02-27: non-SPD `p=1` main-path adaptive toggle

Decision:
- Reject and archive; keep main `PE-Quad-Coupled-Apply` with adaptive toggle off by default.
- Revert temporary benchmark wiring that enabled this toggle.

Why tested:
- We evaluated enabling the existing in-kernel adaptive step selection for the main method
  as a one-change policy switch.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --ab-extra-args-a="--no-p1-use-adaptive-main" --ab-extra-args-b="--p1-use-adaptive-main" --ab-label-a baseline --ab-label-b adaptive_main --ab-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_adaptive_main_step5/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_adaptive_main_step5/manifest.json`

Key results:
- Clear speed regression in all cells (`~+6%` to `~+82%` total ms).
- Accuracy unchanged in all matched rows (`relerr_ratio(B/A)=1.000`).
- Conclusion: strict regression, so this switch is rejected and archived.

## 2026-02-27: SPD `p=1` Chebyshev method exposure in benchmark suite

Decision:
- Keep: expose `Chebyshev-Apply` as an available method for `p=1` in the SPD solve benchmark CLI.
- Do not change default benchmark method selection (`PE-Quad-Coupled-Apply` remains default).

Why tested:
- The p=1 notes propose Chebyshev/minimax polynomial evaluation as a robust SPD path.
- We added only method availability and performed cross-method A/B.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "_spd_p1_klt_n_" --ab-extra-args-a="--methods PE-Quad-Coupled-Apply" --ab-extra-args-b="--methods Chebyshev-Apply" --ab-label-a pe_quad --ab-label-b chebyshev --no-ab-match-on-method --ab-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_vs_pe_step6/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_vs_pe_step6/manifest.json`

Key results:
- Speed: Chebyshev was faster in `5/6` cells (`-2.2%` to `-63.6%`), slower in `1/6` (`+14.5%` on `illcond_1e6, k=64`).
- Accuracy: Chebyshev relerr was higher in `5/6` cells (about `1.14x` to `1.27x` of PE), better in `1/6`.
- Conclusion: mixed speed/accuracy tradeoff, so this remains an optional method for analysis, not a default replacement.

## 2026-02-27: SPD `p=1` Chebyshev mode (`fixed` vs `minimax-auto`)

Decision:
- Keep `--cheb-mode fixed` as the recommended mode for SPD `p=1` benchmark usage.
- Reject `minimax-auto` as a default for this slice.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "_spd_p1_klt_n_" --ab-extra-args-a="--methods Chebyshev-Apply --cheb-mode fixed" --ab-extra-args-b="--methods Chebyshev-Apply --cheb-mode minimax-auto" --ab-label-a cheb_fixed --ab-label-b cheb_minimax_auto --ab-match-on-method --ab-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_mode_step7/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_mode_step7/manifest.json`

Key results:
- Mixed and mostly slower for `minimax-auto`: large regressions in most cells, including a severe outlier (`+176.7%` at `k=64, gaussian_spd`).
- Accuracy differences were negligible (near-parity relerr ratios around `1.0`).
- Conclusion: no performance justification to switch from fixed mode in this p=1 slice.

## 2026-02-27: SPD `p=1` Chebyshev k<n degree cap (`24` vs `16`)

Decision:
- No default change; keep current `--cheb-degree-klt 24` baseline for p=1 benchmarking.
- Treat this run as inconclusive due highly mixed per-case outcomes.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "_spd_p1_klt_n_" --ab-extra-args-a="--methods Chebyshev-Apply --cheb-mode fixed --cheb-degree-klt 24" --ab-extra-args-b="--methods Chebyshev-Apply --cheb-mode fixed --cheb-degree-klt 16" --ab-label-a cheb_klt24 --ab-label-b cheb_klt16 --ab-match-on-method --ab-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_klt_step8/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_klt_step8/manifest.json`

Key results:
- Mixed with conflicting outliers: one cell regressed heavily while another improved heavily.
- Relerr stayed near parity (ratios close to `1.0`) but timing stability was not reliable enough for policy change.
- Conclusion: do not switch k<n cap from `24` to `16` on this evidence.

## 2026-02-27: SPD `p=1` Chebyshev CUDA graph replay (`off` vs `on`)

Decision:
- Keep `--cheb-cuda-graph` enabled for the optional SPD `p=1` Chebyshev path.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "_spd_p1_klt_n_" --ab-extra-args-a="--methods Chebyshev-Apply --no-cheb-cuda-graph" --ab-extra-args-b="--methods Chebyshev-Apply --cheb-cuda-graph" --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-match-on-method --ab-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_cuda_graph_step9/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_spd_p1_cheb_cuda_graph_step9/manifest.json`

Key results:
- Strict speed win in all `6/6` cells (total ms deltas from `-13.5%` to `-66.0%`).
- Relative error unchanged in all matched cells (`relerr_ratio(B/A)=1.000`).
- Conclusion: strong keep decision for CUDA-graph replay when using Chebyshev in this slice.

## 2026-02-27: SPD `p=1` policy compare (`PE-Quad` vs Chebyshev+graph)

Decision:
- Keep current default policy (`PE-Quad-Coupled-Apply`) for SPD `p=1`.
- Keep Chebyshev+graph as an optional contender method.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "_spd_p1_klt_n_" --ab-extra-args-a="--methods PE-Quad-Coupled-Apply" --ab-extra-args-b="--methods Chebyshev-Apply --cheb-cuda-graph" --ab-label-a pe_quad --ab-label-b cheb_graph_on --no-ab-match-on-method --ab-out benchmark_results/runs/2026_02_27/ab_spd_p1_policy_cheb_graph_step10/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_spd_p1_policy_cheb_graph_step10/manifest.json`

Key results:
- Mixed tradeoff: Chebyshev+graph faster in `5/6` cells but slower in one (`gaussian_spd, k=1`).
- Accuracy was usually higher error for Chebyshev (`~1.14x` to `1.27x` relerr in most cells), though one cell improved.
- Conclusion: not a strict default replacement despite strong speed potential; keep as optional method.

## 2026-02-27: non-SPD `p=1` safety guards (`on` vs `off`)

Decision:
- Reject disabling safety guards as default.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --ab-extra-args-a="--nonspd-safe-fallback-tol 0.01 --nonspd-safe-early-y-tol 0.8" --ab-extra-args-b="--nonspd-safe-fallback-tol 0 --nonspd-safe-early-y-tol 0" --ab-label-a safe_on --ab-label-b safe_off --ab-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_safe_guards_step11/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_safe_guards_step11/manifest.json`

Key results:
- Speed improved substantially with guards off.
- But hard-case robustness collapsed (`similarity_posspec_hard` relerr hit `1.000e+00`).
- Conclusion: guards are required for correctness robustness and must stay enabled.

## 2026-02-27: non-SPD `p=1` final safety check diagonal gate

Decision:
- Reject and revert this optimization attempt.

Why tested:
- We tried gating the expensive final residual fallback check with a cheap diagonal proxy:
  skip final residual check when `max|diag(Y)-1|` is below a threshold.

Benchmark arguments:
- `uv run python benchmarks/run_benchmarks.py --only "non-SPD p=1 k<n" --ab-extra-args-a="--nonspd-safe-fallback-tol 0.01 --nonspd-safe-early-y-tol 0.8 --nonspd-safe-final-diag-tol 0" --ab-extra-args-b="--nonspd-safe-fallback-tol 0.01 --nonspd-safe-early-y-tol 0.8 --nonspd-safe-final-diag-tol 0.2" --ab-label-a safe_baseline --ab-label-b safe_diag_gated --ab-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_safe_diag_gate_step12/report.md --manifest-out benchmark_results/runs/2026_02_27/ab_nonspd_p1_safe_diag_gate_step12/manifest.json`

Key results:
- Mixed and often slower; no consistent speed benefit.
- Accuracy stayed near parity, but performance variance/regressions were not acceptable.
- Conclusion: no strict win, so the code change was reverted and archived.

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
- Reject staged candidate (`greedy-minimax" early + forced inverse-Newton tail).
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
