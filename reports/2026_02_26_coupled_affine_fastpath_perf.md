# Coupled Affine Fast-Path Perf Check

*Date: 2026-02-26*

## Scope

- Code path: `fast_iroot/coupled.py` (`inverse_solve_pe_quadratic_coupled`, affine-step updates)
- Objective: reduce affine-step overhead without changing schedule selection or solver API.
- Suite target: `PE-Quad-Coupled-Apply` with default `--online-coeff-mode greedy-affine-opt`.

## What Changed

- Added affine-specialized apply/update helpers to avoid explicit `B = aI + bY` materialization in hot paths.
- Added p-specific affine `Y` updates for:
  - `p=1`: `Y <- aY + bY^2` (single GEMM + fused combine)
  - `p=2`: `Y <- a^2Y + 2abY^2 + b^2Y^3` (two GEMMs + fused combine)
- Integrated these fast paths in both:
  - `inverse_proot_pe_quadratic_coupled`
  - `inverse_solve_pe_quadratic_coupled`

## Correctness

- `uv run python -m pytest -q`
- Result: `57 passed`

## Profiling (Operator-Level)

Profile setup: CUDA, bf16, `n=1024`, `k=64`, `p=2`, affine-opt schedule, repeated coupled-solve calls.

From:
- `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/profile/baseline_top_ops.txt`
- `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/profile/optimized_top_ops.txt`

Observed shifts:
- `Self CUDA time total`: `18.295ms -> 17.910ms` (`-2.1%`)
- `aten::copy_`: `1.711ms (48 calls) -> 1.516ms (40 calls)`
- `aten::mm`: `9.676ms (80 calls) -> 9.552ms (72 calls)`
- `aten::addmm`: `5.170ms (24 calls) -> 5.329ms (32 calls)`

Interpretation: the change reduced explicit copy/mm volume and shifted work toward fused `addmm`, which lowered total self CUDA time in this profiled case.

## End-to-End Suite (Same Interpreter)

Harness: `matrix_solve.py`, `n=1024`, `p in {1,2,4}`, `k in {1,16,64}`, cases `{gaussian_spd, illcond_1e6}`, `trials=20`, `timing_reps=5`, `bf16`, `precond=jacobi`.

Artifacts:
- Baseline logs: `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/baseline_sameenv/`
- Optimized logs: `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/optimized_sameenv/`
- Parsed table: `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/summary_coupled_apply_fastpath_sameenv.md`

Key deltas (`optimized vs baseline`, means over 18 cells):
- `iter_ms`: **-1.04%**
- `relerr`: **-3.15%**
- `total_ms`: `+4.38%` (dominated by preconditioning variance outside optimized kernel path)

## Kernel-Focused Microbenchmark

Artifacts:
- `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/microbench_baseline.csv`
- `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/microbench_optimized.csv`
- `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/microbench_compare.md`

Setup: fixed `A_norm`, timed only `inverse_solve_pe_quadratic_coupled` via CUDA events.

Overall delta:
- median per-call time: **-0.82%**
- mean per-call time: **-1.01%**
- Note: one skinny-RHS cell (`p=1, k=1`) regressed in median timing, while most other cells improved or were neutral.

## Conclusion

- The affine fast-path refactor is a small net kernel-path win (about `~1%` on isolated coupled-solve timing, and reduced profiler self-CUDA time in the targeted affine-heavy case), with mixed per-cell impact at very small RHS width.
- No API changes or schedule-policy changes were introduced.
