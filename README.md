# Fast Matrix Inverse p-th Roots (GPU-Focused)

Fast, practical inverse p-th root iteration for SPD matrices, tuned for ML preconditioning workloads.

This project prioritizes:
- fixed small iteration budgets
- GEMM-dominated kernels (matmul-only, no solves/QR)
- bf16-friendly stability
- empirical benchmarking over purely theoretical comparisons

## Repository Layout

- `fast_iroot/`
  - `precond.py` — Preconditioning logic (`precond_spd`)
  - `coupled.py` — Coupled quadratic PE iterations (`inverse_sqrt_pe_quadratic`, `inverse_proot_pe_quadratic_coupled`)
  - `uncoupled.py` — Uncoupled quadratic PE iterations (`inverse_proot_pe_quadratic_uncoupled`)
  - `chebyshev.py` — Chebyshev Minimax Polynomial Direct Apply logic (`apply_inverse_proot_chebyshev`)
  - `apply.py` — Direct apply inverse / inverse-root wrappers.
  - `coeffs.py` — Coefficient schedule loading/tuning hooks (`build_pe_schedules`)
  - `metrics.py` — Quality metrics (`compute_quality_stats`, `exact_inverse_proot`)
  - `utils.py` — Low-level helpers (`_matmul_into`, `_addmm_into`, `_bpow_times_y`)
  - `auto_policy.py` — Legacy auto-policy utilities (currently unused)
- `matrix_iroot.py`
  - Main benchmark harness CLI for explicit inverse p-th roots
- `matrix_solve.py`
  - Benchmark harness CLI tailored for Direct Chebyshev Solves ($Z \approx A^{-1/p} B$) 
- `coeff_tuner.py`
  - Offline schedule tuning utility
- `verify_iroot.py`
  - Correctness test across p∈{1,2,3,4,8}
- `reports/`
  - Benchmark results and comprehensive report (`chebyshev_solve_benchmark.md`)
- `archive/`
  - Archived affine/NS methods (deprecated, reference only)

## Features & Usage

### 1. Generating $X \approx A^{-1/p}$
Constructing explicit precision dense roots ($O(N^3)$ operation).

```python
import torch
from fast_iroot import precond_spd, inverse_proot_pe_quadratic_uncoupled

A = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
A = (A @ A.mT) / 1024

# 1. Precondition the SPD matrix (spectral scaling)
A_norm, stats = precond_spd(A, mode="frob", l_target=0.05)

# 2. Iterate inverse square root explicitly (N x N)
X_norm, _ = inverse_proot_pe_quadratic_uncoupled(
    A_norm,
    abc_t=pe_schedule, # Generate with `build_pe_schedules`
    p_val=2
)
```

### 2. Direct Solve $Z \approx A^{-1/p} B$
Leverages PyTorch-native Clenshaw recurrences over precomputed minimax polynomials. Perfect for whitening feature sets / preconditioning gradients where $K \ll N$, resulting in $O(N^2 K)$ memory and operational complexity ($> 10 \times$ speedup vs Dense Matrix formation).

```python
import torch
from fast_iroot import precond_spd, apply_inverse_proot_chebyshev

# Given an SPD Matrix A (N x N) and an RHS Matrix B (N x K)
A = torch.randn(4096, 4096, dtype=torch.float32, device="cuda"); A = A @ A.mT
B = torch.randn(4096, 32, dtype=torch.float32, device="cuda")

A_norm, stats = precond_spd(A, mode="frob", l_target=0.05)

# Direct Clenshaw solver avoiding N x N memory footprints
Z, _ = apply_inverse_proot_chebyshev(
    A=A_norm,
    B=B,
    p_val=2, 
    degree=32, 
    l_min=0.05
)
```

## Environment

`pyproject.toml` is configured for `uv` and CUDA-enabled PyTorch wheels.

### Install

```bash
uv sync
```

## Quick Start

Run a quick benchmark (inverse 4th root):

```bash
uv run python matrix_iroot.py --p 4 --sizes 256,512 --dtype bf16 --trials 8
```

Run for matrix inverse (p=1):

```bash
uv run python matrix_iroot.py --p 1 --sizes 256,512,1024 --dtype bf16 --trials 8 --coeff-mode tuned
```

Verify correctness across multiple p values:

```bash
uv run python verify_iroot.py
```

## Methods

The project uses **quadratic polynomial-express (PE-Quad)** iterations exclusively:

### PE-Quad (Uncoupled)
Tracks only `X ≈ A^{-1/p}`, recomputing `Y = X^p · A` each step.
- Lower memory (5 workspace tensors)
- Works for any p

### PE-Quad-Coupled
Tracks both `X ≈ A^{-1/p}` and `Y ≈ A · X^p`.
- Terminal-step optimization: skips Y-update on last iteration (saves 2-3 matmuls)
- 10-14% faster iteration time for p≥2 at larger sizes
- Works for any p via binary exponentiation and similarity transforms (`_bpow`)

### Deprecated Methods (archived)
Affine methods (PE-Affine, Newton-Schulz NS3/NS4, PE-NS3) are archived in `archive/affine_iterations.py`. They consistently underperform quadratic methods in both speed and residual quality.

## Important CLI Flags

```text
--p               Root exponent (1=inverse, 2=inv sqrt, 4=inv 4th root, etc.)
--sizes           Matrix dimensions to benchmark
--dtype {fp32,bf16}
--trials          Number of test matrices per case
--compile         Enable torch.compile
--precond {none,frob,aol}
--coeff-mode {auto,precomputed,tuned}
--coeff-seed      Seed for coefficient tuning
--coeff-safety    Safety scaling factor
--target-resid    Target residual threshold
--metrics-mode {full,fast}
--power-iters     Spectral residual estimation iterations
--mv-samples      Random MV probe sample count
--hard-probe-iters  Hard-direction probe iterations
```

## Metrics Reported

Per method and case:
- Total median ms (precond + iteration)
- Iteration median ms
- Residual (median / p95 / max)
- Relative error vs eigendecomp
- Symmetry diagnostics (symX, symW)
- Bad count (NaN/Inf)

## Results

See `results/benchmark_report.md` for the latest comprehensive benchmark data.

## Tuning Coefficients

Use `coeff_tuner.py` for offline schedule generation:
- Precomputed schedules available for p=2, l_target=0.05
- Tuned schedules for arbitrary p and targets via `--coeff-mode tuned`
- Optional safety scaling

## References

- Amsel et al., 2025. *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm* (arXiv:2505.16932)
- Boissin et al., 2025. *Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning* (arXiv:2512.04632)
