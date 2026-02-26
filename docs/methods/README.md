# Method Documentation

This section documents active methods used by the solver/apply workflow.

## Active Methods

- `docs/methods/shared_tricks.md`
  - Shared math model, preconditioning pipeline, stability tricks
- `docs/methods/pe2.md`
  - Quadratic PE iteration (PE-Quad) — the primary method
- `docs/methods/uncoupled_p_root.md`
  - Uncoupled formulation for general p-th roots
- `docs/methods/chebyshev.md`
  - Direct Clenshaw evaluation for $X = A^{-1/p} B$ without dense inversions

## Archived (Deprecated)

Affine/NS methods have been archived to `archive/` as they consistently
underperform quadratic methods:
- `archive/ns.md` — Newton-Schulz (NS3, NS4)
- `archive/pe_ns3.md` — Affine PE schedule (PE-NS3)
- `archive/auto.md` — AUTO selection policy
- `archive/affine_iterations.py` — All affine iteration code

## Source of Truth

- Core kernels: `fast_iroot/`
- Benchmark harnesses: `benchmarks/run_benchmarks.py`, `benchmarks/solve/matrix_solve.py`, `benchmarks/solve/matrix_solve_nonspd.py`
- Benchmark cores: `benchmarks/solve/bench_solve_core.py`
- Quality metrics: `fast_iroot/metrics.py`

