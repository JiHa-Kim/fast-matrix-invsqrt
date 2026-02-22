# Matrix Inverse Square Root (GPU-Focused, Benchmark-Driven)

Fast, practical inverse square root iteration for SPD matrices, tuned for ML preconditioning workloads.

This project prioritizes:
- fixed small iteration budgets
- GEMM-dominated kernels
- bf16-friendly stability
- empirical benchmarking over purely theoretical comparisons

## Repository Layout

- `isqrt_core.py`
  - Preconditioning (`precond_spd`)
  - Iteration kernels (`inverse_sqrt_ns`, `inverse_sqrt_pe_affine`, `inverse_sqrt_pe_quadratic`)
  - AUTO policy utilities (`AutoPolicyConfig`, `choose_auto_method`)
  - Coefficient schedule loading/tuning hooks (`build_pe_schedules`)
- `isqrt_metrics.py`
  - Quality metrics (residual, symmetry, spectral proxy, apply-to-vector error)
  - Optional high-cost relative error helper
- `matrix_isqrt.py`
  - Main benchmark harness CLI
  - Case generation, timing, reporting, and method comparison
- `coeff_tuner.py`
  - Offline schedule tuning utility
- `artifacts/benchmarks/`
  - Raw benchmark outputs
- `reports/`
  - Human-readable benchmark reports and extracted summaries

## Environment

`pyproject.toml` is configured for `uv` and CUDA-enabled PyTorch wheels.

### Install (recommended)

```bash
uv sync
```

If needed:

```bash
uv venv .venv
uv pip install -e .
```

## Quick Start

Run a quick benchmark:

```bash
python matrix_isqrt.py --sizes 256,512 --dtype bf16 --trials 8 --warmup 2 --metrics-mode fast
```

Run a rigorous benchmark:

```bash
python matrix_isqrt.py --sizes 256,512,1024 --dtype bf16 --trials 30 --warmup 4 --timing-reps 80 --metrics-mode fast --power-iters 8 --mv-samples 8
```

## Methods Compared

`matrix_isqrt.py` benchmarks:
- `NS3`
- `NS4`
- `PE-NS3` (affine schedule)
- `PE2` (quadratic schedule)
- `AUTO` (policy-based choice)

### Current Main-Path Defaults

The main runner is intentionally isolated to the favorable runtime path:
- row-sum based normalization in preconditioning
- terminal final-step optimization enabled in kernels

Legacy/slower variants are not exposed in the primary benchmark CLI.

## Important CLI Flags

```text
--sizes
--dtype {fp32,bf16}
--trials
--warmup
--timing-reps
--precond {none,frob,aol}
--ridge-rel
--l-target
--target-resid
--target-metric {residual,hard_dir}
--auto-policy {size_rho,interval,hybrid}
--kappa-ns3-max
--kappa-pe2-min
--coeff-mode {auto,precomputed,tuned}
--coeff-seed
--coeff-safety
--coeff-no-final-safety
--metrics-mode {full,fast}
--power-iters
--mv-samples
--hard-probe-iters
```

## Metrics Reported

Per method and case:
- total median ms
- preconditioning median ms
- iteration median ms
- median residual
- p95 residual
- max residual
- optional spectral residual proxy (`--power-iters`)
- optional hard-direction residual probe (`--hard-probe-iters`)
- symmetry diagnostics (`symX`, `symW`)
- apply-to-vector proxy (`--mv-samples`)
- bad count (NaN/Inf)

## Results Organization

- Raw outputs: `artifacts/benchmarks/`
- Current benchmark report: `reports/benchmark_report_2026-02-21.md`
- Less favorable methods archive: `reports/less_favorable_methods.md`

## Tuning Coefficients

Use `coeff_tuner.py` for offline schedule generation. The benchmark harness supports:
- precomputed schedules (default for `l_target=0.05`)
- tuned schedules for non-default targets
- optional safety scaling

## Notes

- This code is oriented to optimizer preconditioning use-cases where stable, fast approximate inverse square root is preferred over exact high-precision linear algebra.
- For throughput studies, prefer `--metrics-mode fast`; use `full` selectively for spot checks.

## References

- Amsel et al., 2025. *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm* (arXiv:2505.16932)
- Boissin et al., 2025. *Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning* (arXiv:2512.04632)
- Newton-Schulz background: https://docs.modula.systems/algorithms/newton-schulz/
