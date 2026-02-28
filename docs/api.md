# API Reference

This page documents the high-level production API of `fast_iroot`. For low-level kernel details, see the [Methods](methods/README.md) section.

## Configuration Objects

### `ScheduleConfig`

Configuration for building a quadratic **Polynomial-Express (PE)** coefficient schedule.

```python
@dataclass(frozen=True)
class ScheduleConfig:
    l_target: float = 0.05
    coeff_mode: str = "auto"
    coeff_seed: int = 0
    coeff_safety: float = 1.0
    coeff_no_final_safety: bool = False
```

- **`l_target`**: Lower spectral bound target for the schedule (e.g., $0.05$ assumes eigenvalues in $[0.05, 1]$).
- **`coeff_mode`**: Strategy for selecting coefficients (`"auto"`, `"greedy-newton"`, `"greedy-minimax"`, etc.).
- **`coeff_safety`**: Global multiplier for the contraction interval.

### `PrecondConfig`

Configuration for **Symmetric Positive Definite (SPD)** preconditioning used by `solve_spd`.

```python
@dataclass(frozen=True)
class PrecondConfig:
    mode: str = "none"
    eps: float = 1e-12
    ruiz_iters: int = 2
    ridge_rel: float = 0.0
    l_target: float = 0.05
    lambda_max_est: str = "row_sum"
    lambda_max_power_iters: int = 8
    lambda_max_safety: float = 1.02
```

- **`mode`**: Preconditioning mode (`"none"`, `"jacobi"`, `"ruiz"`, `"aol"`, `"frob"`).
- **`l_target`**: Target lower bound for the spectral interval after preconditioning.
- **`lambda_max_est`**: Strategy for estimating $\lambda_{max}$ (`"row_sum"`, `"power"`).

---

## High-Level Solvers

### `solve_spd`

Primary entrypoint for solving $Z \approx A^{-1/p} B$ where $A$ is **Symmetric Positive Definite (SPD)**.

```python
def solve_spd(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    p_val: int = 2,
    abc_t: Optional[torch.Tensor] = None,
    schedule_config: Optional[ScheduleConfig] = None,
    precond_config: Optional[PrecondConfig] = None,
    workspace: Optional[InverseApplyAutoWorkspace] = None,
    **kwargs
) -> Tuple[torch.Tensor, InverseApplyAutoWorkspace, PrecondStats, str]:
```

- **Returns**: `(Z, workspace, stats, schedule_desc)`.
- **Note**: Supports $p=2$ (inverse sqrt), $p=4$, and general $p$ via coupled iterative kernels.

### `solve_nonspd`

Entrypoint for solving $Z \approx A^{-1} B$ where $A$ is a general non-SPD matrix.

```python
def solve_nonspd(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    p_val: int = 1,
    schedule_config: Optional[ScheduleConfig] = None,
    nonspd_precond_mode: str = "row-norm",
    **kwargs
) -> Tuple[torch.Tensor, InverseApplyAutoWorkspace, str]:
```

- **Note**: Currently restricted to $p=1$ (standard linear solve). Use `solve_spd` for $p > 1$ if the matrix is SPD.

### `solve_gram_spd`

Specialized entrypoint for Gram-matrix solve $Z \approx (G^T G)^{-1/p} B$.

```python
def solve_gram_spd(
    G: torch.Tensor,
    B: torch.Tensor,
    *,
    p_val: int = 2,
    reuse_precond: bool = True,
    **kwargs
) -> Tuple[torch.Tensor, GramInverseApplyWorkspace, PrecondStats, str]:
```

- **Efficiency**: Avoids explicit formation of $G^T G$ if $k \ll n$ or if using matrix-free Chebyshev paths.
- **`reuse_precond`**: If `True`, caches the preconditioner (e.g., column norms) for subsequent calls with the same `G`.

---

## Utility Functions

### `build_schedule`

Builds a PE-Quadratic coefficient schedule.

```python
def build_schedule(
    device: torch.device,
    *,
    p_val: int = 2,
    config: Optional[ScheduleConfig] = None,
) -> Tuple[torch.Tensor, str]:
```

Returns the coefficients `abc_t` and a description string.
