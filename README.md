# Fast Matrix Inverse p-th Roots

GPU-focused inverse p-th-root kernels with a solver-first benchmark workflow.

## Core Focus

- Direct solve/apply path (`Z = A^{-1/p} B`) is the primary production path.
- Explicit inverse-root materialization (`X = A^{-1/p}`) is secondary.

## Repository Layout

- `fast_iroot/`: core library kernels and preconditioning utilities.
- `benchmarks/solve/`: primary solver benchmarks.
  - `matrix_solve.py` (SPD)
  - `matrix_solve_nonspd.py` (non-SPD, p=1)
  - `ablate_solve_inverse_ideas.py`
- `benchmarks/common.py`: shared benchmark helpers.
- `tests/`: pytest suite.
- `benchmark_results/`: raw benchmark logs.
- `reports/latest/`: current summaries generated from fresh runs.

## Install

```bash
uv sync
```

## Verification

```bash
uv run python -m pytest -q
uv run python -m pytest tests/test_verify_iroot.py -q
```

## Main Benchmark Entry Point

```bash
uv run python benchmarks/run_benchmarks.py
```

This runs the maintained batch benchmark flow and writes logs under `benchmark_results/`.

## Direct Solver Benchmark Commands

```bash
uv run python -m benchmarks.solve.matrix_solve --p 1 --sizes 1024 --k 1,16,64,1024 --trials 5 --dtype bf16
uv run python -m benchmarks.solve.matrix_solve_nonspd --p 1 --sizes 1024 --k 1,16,64,1024 --trials 5 --dtype bf16
```

## Notes

- Newton baselines are included in solve benchmarks:
  - `Inverse-Newton-Inverse-Multiply`
  - `Inverse-Newton-Coupled-Apply`
- Report files are expected to be regenerated from fresh benchmark logs when needed.
