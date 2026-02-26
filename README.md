# Fast Matrix Inverse p-th Roots (GPU-Focused)

Practical inverse p-th-root kernels for SPD matrices, optimized for fixed-iteration, GEMM-heavy workloads in ML preconditioning.

## What This Repo Provides

- Explicit inverse p-th roots:
  - `inverse_proot_pe_quadratic_uncoupled`
  - `inverse_proot_pe_quadratic_coupled`
- Direct apply to RHS blocks (`Z = A^{-1/p} B`) without materializing dense `A^{-1/p}`:
  - `apply_inverse_proot_chebyshev`
  - `inverse_solve_pe_quadratic_coupled`
- Preconditioning + diagnostics:
  - `precond_spd`
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
  - Narrative benchmark notes (including Chebyshev solve report).
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
uv run python scripts/matrix_solve.py --p 2 --sizes 1024,2048 --k 16 --trials 3 --timing-reps 5 --dtype bf16 --precond frob --l-target 0.05 > artifacts/benchmarks/solve_p2_k16_2026-02-25.txt
uv run python scripts/matrix_solve.py --p 2 --sizes 1024,2048 --k 64 --trials 3 --timing-reps 5 --dtype bf16 --precond frob --l-target 0.05 > artifacts/benchmarks/solve_p2_k64_2026-02-25.txt
```

## Key CLI Flags

- `--p`: root exponent.
- `--sizes`: comma-separated matrix sizes.
- `--dtype {fp32,bf16}`.
- `--precond {none,frob,aol}`.
- `--coeff-mode {auto,precomputed,tuned}` (inverse-root harness).
- `--compile`: enable `torch.compile`.
- `--timing-reps`: average repeated runs per trial.
- `--symmetrize-every`: symmetrize cadence for coupled `Y`.
- `--metrics-mode {full,coupled}` (inverse-root harness).

## Latest Benchmark Artifacts

- Inverse-root report: `results/benchmark_report.md` (generated 2026-02-25).
- Solve/apply narrative: `reports/chebyshev_solve_benchmark.md` (updated from 2026-02-25 raw logs).
- Solve raw logs:
  - `artifacts/benchmarks/solve_p2_k16_2026-02-25.txt`
  - `artifacts/benchmarks/solve_p2_k64_2026-02-25.txt`

## References

- Guo & Higham (2006), *A Schur-Newton Method for the Matrix p-th Root and its Inverse*: https://eprints.maths.manchester.ac.uk/850/
- Amsel et al. (2025), *The Polar Express*: https://arxiv.org/abs/2505.16932
- Li et al. (CVPR 2018), *iSQRT-COV*: https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html
