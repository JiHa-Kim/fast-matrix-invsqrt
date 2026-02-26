# Fast Matrix Inverse p-th Roots (GPU-Focused)

Practical inverse p-th-root kernels for SPD matrices, optimized for fixed-iteration, GEMM-heavy workloads in ML preconditioning.

## What This Repo Provides

- Explicit inverse p-th roots:
  - `inverse_proot_pe_quadratic_uncoupled`
  - `inverse_proot_pe_quadratic_coupled`
- Direct apply to RHS blocks (`Z = A^{-1/p} B`) without materializing dense `A^{-1/p}`:
  - `apply_inverse_proot_chebyshev`
  - `inverse_solve_pe_quadratic_coupled`
  - `apply_inverse_root_auto` (single-shot vs reuse-aware strategy)
- Preconditioning + diagnostics:
  - `precond_spd`
  - `precond_gram_spd` (`A = G^T G` path)
  - `compute_quality_stats`, `iroot_relative_error`

## Repository Layout

- `fast_iroot/`
  - Core kernels and utilities.
- `scripts/`
  - `matrix_iroot.py`: inverse-root benchmark CLI.
  - `matrix_solve.py`: solve/apply benchmark CLI.
  - `verify_iroot.py`: correctness/stability sweep.
  - `generate_benchmark_report.py`: regenerates `results/benchmark_report.md`.
- `scripts/bench_common.py`, `scripts/bench_iroot_core.py`, `scripts/bench_solve_core.py`
  - Shared benchmark engines/helpers.
- `results/`
  - Latest generated inverse-root benchmark report.
- `reports/`
  - Narrative benchmark notes.
- `artifacts/benchmarks/`
  - Raw benchmark logs used to build reports.
- `docs/methods/`
  - Method-level docs.

## Install

```bash
uv sync
```

## Quick Verification

```bash
uv run python -m pytest -q
uv run python scripts/verify_iroot.py
```

## Benchmark Commands

Regenerate inverse-root benchmark report (compiled, p in `{1,2,3,4,8}`):

```bash
uv run python scripts/generate_benchmark_report.py --out results/benchmark_report.md --sizes 256,512,1024 --trials 10
```

Reproduce latest solve/apply benchmark logs:

```bash
uv run python scripts/matrix_solve.py --p 2 --sizes 1024,2048 --k 16 --trials 3 --timing-reps 5 --dtype bf16 --precond jacobi --l-target 0.05 > artifacts/benchmarks/solve_p2_k16_2026-02-25.txt
uv run python scripts/matrix_solve.py --p 2 --sizes 1024,2048 --k 64 --trials 3 --timing-reps 5 --dtype bf16 --precond jacobi --l-target 0.05 > artifacts/benchmarks/solve_p2_k64_2026-02-25.txt
```

## Key CLI Flags

- `--p`: root exponent.
- `--sizes`: comma-separated matrix sizes.
- `--dtype {fp32,bf16}`.
- `--precond {none,frob,aol,jacobi,ruiz}`.
- `--precond-ruiz-iters`: equilibration rounds for `ruiz`.
- `--coeff-mode {auto,precomputed,tuned}` (inverse-root harness).
- `--compile`: enable `torch.compile`.
- `--timing-reps`: average repeated runs per trial.
- `--symmetrize-every`: symmetrize cadence for coupled `Y`.
- `--online-coeff-mode {off,greedy-newton,greedy-minimax,greedy-affine-opt}`: optional cost-aware per-step PE schedule adaptation for coupled apply (`greedy-affine-opt` is default).
- `--online-coeff-min-rel-improve`: switch threshold for `--online-coeff-mode=greedy-newton`.
- `--online-coeff-min-ns-logwidth-rel-improve`: minimax-vs-NS dominance margin for `--online-coeff-mode=greedy-minimax`.
- `--online-stop-tol`, `--online-min-steps`: low-overhead coupled early-stop controls.
- `--metrics-mode {full,coupled}` (inverse-root harness).

## Latest Benchmark Artifacts

- Inverse-root report: `reports/2025_02_25_benchmark_p1thru5.md`.
- Solve online-coefficient ablation (`20` trials): `reports/2026_02_25_solve_online_coeff_ablation_t20.md`.
- Affine online-schedule ablation (`20` trials): `reports/2026_02_26_affine_online_coeff_ablation_t20.md`.
- Coupled affine fast-path perf check: `reports/2026_02_26_coupled_affine_fastpath_perf.md`.
- Square-RHS direct-vs-materialize validation (`ideas/3`): `reports/2026_02_25_idea3_square_rhs_apply_vs_materialize.md`.
- Preconditioner ablation + Gram path checks (`ideas/4`): `reports/2026_02_26_precond_and_gram_path_ablation.md`.
- Raw logs:
  - `benchmark_results/2026_02_25/solve_ablation_t20/`
  - `benchmark_results/2026_02_25/solve_exploratory/`
  - `benchmark_results/2026_02_25/iroot_p1_p5/`
  - `benchmark_results/2026_02_26/idea4_precond_t20/`
  - `benchmark_results/2026_02_26/idea4_precond_iroot_t20/`
  - `benchmark_results/2026_02_26/idea4_gram_precond_t20/`
  - `benchmark_results/2026_02_26/idea_affine_online_t20/`
  - `benchmark_results/2026_02_26/perf_coupled_affine_fastpath_t20/`

## References

- Guo & Higham (2006), *A Schur-Newton Method for the Matrix p-th Root and its Inverse*: https://eprints.maths.manchester.ac.uk/850/
- Amsel et al. (2025), *The Polar Express*: https://arxiv.org/abs/2505.16932
- Li et al. (CVPR 2018), *iSQRT-COV*: https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html
