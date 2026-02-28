# fast-matrix-inverse-roots

Production-oriented inverse p-th-root and inverse-apply kernels for PyTorch workloads.

`fast_iroot` focuses on the path that matters most in ML systems: directly computing
`Z ~= A^(-1/p) B` without materializing dense inverse operators unless reuse demands it.

## Highlights

- Coupled quadratic PE kernels for inverse and inverse-root apply.
- SPD and non-SPD paths with explicit preconditioning utilities.
- Gram-matrix workflows for common `G^T G` and `G G^T` patterns.
- Workspace-aware APIs to reduce allocation overhead in repeated calls.
- Benchmark harnesses and reproducible manifests under `benchmark_results/runs/`.

## Installation

```bash
uv sync
```

## Quickstart

### 1) SPD solve/apply with high-level API

```python
import torch
from fast_iroot import solve_spd, PrecondConfig, ScheduleConfig

n, k = 256, 16
A = torch.randn(n, n)
A = A.mT @ A + 1e-2 * torch.eye(n)
B = torch.randn(n, k)

Z, ws, stats, schedule = solve_spd(
    A,
    B,
    p_val=2,
    precond_config=PrecondConfig(mode="jacobi", l_target=0.05),
    schedule_config=ScheduleConfig(coeff_mode="auto"),
)

print(Z.shape, stats.kappa_proxy, schedule)
```

### 2) Gram SPD path

```python
import torch
from fast_iroot import solve_gram_spd

m, n, k = 512, 128, 32
G = torch.randn(m, n)
B = torch.randn(n, k)

Z, ws, stats, _ = solve_gram_spd(G, B, p_val=2, reuse_precond=True)
```

### 3) Low-level modules (advanced)

```python
from fast_iroot.coeffs import build_pe_schedules
from fast_iroot.precond import precond_spd
from fast_iroot.apply import apply_inverse_root_auto
```

## Documentation

For detailed information on the API, mathematical methods, and benchmarked decisions, see the [Documentation Index](docs/index.md).

## Benchmarks

`fast_iroot` is optimized for ML-sized blocks (e.g., $n=1024$). In production benchmarks, our methods consistently outperform both Vanilla Newton-Schulz and generic PyTorch decompositions:

- **SPD Roots**: Chebyshev and PE-Quad paths are up to **10-15x faster** than dense EVD for inverse-root applies.
- **Gram Matrices**: Specialized dual-paths provide a **>2x speedup** over primal approaches.
- **Quality Assessment**: Reports track median/tail error (`relerr`, `relerr_p90`), failure rate, and quality-per-ms (`q_per_ms`) so method selection is not based on latency alone.

See the [Latest Production Benchmark Report](docs/benchmarks/benchmark_results_production.md) for full details.

## Project Layout

- `fast_iroot/`: kernels, apply paths, preconditioners, diagnostics.
- `benchmarks/`: benchmark drivers and solve suites.
- `tests/`: pytest suite.
- `docs/`: method and implementation documentation. See [docs/index.md](docs/index.md).
- `benchmark_results/`: run artifacts, reports, and manifests.

## License

MIT (`LICENSE`).
