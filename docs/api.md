# API Reference

This package exposes a lean root API plus explicit low-level modules.

- Root package `fast_iroot`: production entrypoints only.
- Submodules `fast_iroot.*`: low-level kernels/workspaces.

## High-Level API

## `ScheduleConfig`

Quadratic PE coefficient schedule settings.

Fields:

- `l_target: float = 0.05`
- `coeff_mode: str = "auto"`
- `coeff_seed: int = 0`
- `coeff_safety: float = 1.0`
- `coeff_no_final_safety: bool = False`

## `PrecondConfig`

SPD (symmetric positive-definite) preconditioner settings used by `solve_spd`.

Fields:

- `mode: str = "none"`
- `eps: float = 1e-12`
- `ruiz_iters: int = 2`
- `ridge_rel: float = 0.0`
- `l_target: float = 0.05`
- `lambda_max_est: str = "row_sum"`
- `lambda_max_power_iters: int = 8`
- `lambda_max_safety: float = 1.02`

## `build_schedule(device, *, p_val=2, config=None)`

Builds a quadratic PE schedule.

Returns:

- `abc_t: torch.Tensor`
- `schedule_desc: str`

## `solve_spd(A, B, *, ...)`

Primary SPD entrypoint for:

`Z ~= A^(-1/p) B`

Pipeline:

1. `precond_spd(A, ...)`
2. `apply_inverse_root_auto(A_norm, B, ...)`

Returns:

- `Z: torch.Tensor`
- `workspace: InverseApplyAutoWorkspace`
- `stats: PrecondStats`
- `schedule_desc: str`

## `solve_nonspd(A, B, *, ...)`

Primary non-SPD entrypoint for:

`Z ~= A^(-1/p) B`

Current scope:

- `p_val=1` only (`Z ~= A^(-1) B`).

Pipeline:

1. `precond_nonspd(A, ...)`
2. `apply_inverse_root_auto(..., assume_spd=False)`

Returns:

- `Z: torch.Tensor`
- `workspace: InverseApplyAutoWorkspace`
- `schedule_desc: str`

## `solve_gram_spd(G, B, *, ...)`

Primary Gram-SPD entrypoint for:

`Z ~= (G^T G)^(-1/p) B`

Pipeline:

1. Cached Gram preconditioning in `apply_inverse_root_gram_spd`
2. Auto strategy selection (`direct-solve`, `materialize-root`, hybrid for p=1)

Returns:

- `Z: torch.Tensor`
- `workspace: GramInverseApplyWorkspace`
- `stats: PrecondStats`
- `schedule_desc: str`

## Low-Level Modules

Import low-level operations from explicit modules:

- `fast_iroot.coeffs`
- `fast_iroot.precond`
- `fast_iroot.apply`
- `fast_iroot.coupled`
- `fast_iroot.uncoupled`
- `fast_iroot.chebyshev`
- `fast_iroot.nsrc`
