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

## Public API

Primary production entrypoints:

- `solve_spd`
- `solve_nonspd`
- `solve_gram_spd`
- `build_schedule`
- `ScheduleConfig`
- `PrecondConfig`

Detailed function-level reference: `docs/api.md`.

Note:

- `solve_nonspd` is currently `p=1` only.

## Validation

```bash
uv run python -m pytest -q
uv run python -m pytest tests/test_verify_iroot.py -q
uv run python -m ruff check .
```

## Benchmarks

Run the maintained benchmark matrix:

```bash
uv run python benchmarks/run_benchmarks.py
```

Generate a consolidated markdown report in a per-run folder:

```bash
uv run python benchmarks/run_benchmarks.py --markdown --out benchmark_results/runs/2026_02_26/full_matrix/report.md --manifest-out benchmark_results/runs/2026_02_26/full_matrix/manifest.json
```

Focused A/B compare:

```bash
uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n" --ab-extra-args-a "--online-coeff-target-interval-err 0.0" --ab-extra-args-b "--online-coeff-target-interval-err 0.01" --ab-out benchmark_results/runs/2026_02_26/ab_spd_p2_klt_n/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_spd_p2_klt_n/manifest.json
```

## Project Layout

- `fast_iroot/`: kernels, apply paths, preconditioners, diagnostics.
- `benchmarks/`: benchmark drivers and solve suites.
- `tests/`: pytest suite.
- `docs/`: method and implementation documentation.
- `benchmark_results/`: run artifacts, reports, and manifests.

## License

MIT (`LICENSE`).
