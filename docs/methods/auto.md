# AUTO Method (Policy-Based Selection)

Benchmark name:

- `AUTO`

Policy definitions:

- `isqrt_core.py:AutoPolicyConfig`
- `isqrt_core.py:choose_auto_method`

## Candidate Methods

AUTO chooses one of:

- `NS3`
- `PE-NS3`
- `PE2`

## Inputs Used for Selection

From preconditioning stats and runtime config:

- matrix size `n`
- `rho_proxy`
- `kappa_proxy`
- policy mode and thresholds:
  - `size_rho`
  - `interval`
  - `hybrid`

## Policies

### `size_rho`

$$
\text{if } (n \ge n_{\text{switch}})\ \lor\ (\rho_{\text{proxy}} \ge \rho_{\text{switch}})\ \Rightarrow\ \text{PE2}
$$

Else:

$$
\text{PE-NS3}
$$

### `interval`

$$
\kappa_{\text{proxy}} \ge \kappa_{\text{pe2,min}} \Rightarrow \text{PE2}
$$

$$
\kappa_{\text{proxy}} \le \kappa_{\text{ns3,max}} \Rightarrow \text{NS3}
$$

Else:

$$
\text{PE-NS3}
$$

### `hybrid`

- starts from size/rho guard
- refines using interval thresholds

## Code Tricks Used

- selection is done once per matrix instance (inside eval loop)
- selected kernel still uses all shared low-level optimizations

## Practical Notes

AUTO is only as good as its proxy thresholds. In this project, benchmark reports are used to tune policy defaults empirically.
