# Method Documentation

This section documents each inverse-square-root method in the project, with the exact math model and implementation details used in code.

## Files

- `docs/methods/shared_tricks.md`
  - shared math model
  - preconditioning pipeline
  - stability and performance tricks used across methods
- `docs/methods/ns.md`
  - Newton-Schulz (`NS3`, `NS4`)
- `docs/methods/pe_ns3.md`
  - Polynomial-Express affine schedule (`PE-NS3`)
- `docs/methods/pe2.md`
  - Polynomial-Express quadratic schedule (`PE2`)
- `docs/methods/auto.md`
  - AUTO selection policy and thresholds

## Source of Truth in Code

- Core kernels and scheduling: `isqrt_core.py`
- Benchmark harness and method comparison: `matrix_isqrt.py`
- Quality metrics and diagnostics: `isqrt_metrics.py`
