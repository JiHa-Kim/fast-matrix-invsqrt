# Solve Online-Coefficient Ablation (20 Trials)

*Date: 2026-02-25*

## Setup
- Workload: `scripts/matrix_solve.py` coupled apply (`PE-Quad-Coupled-Apply`)
- Matrix: `n=1024`, `k in {1,16,64}`, cases `{gaussian_spd, illcond_1e6}`
- Exponents: `p in {1,2,4}`
- Precision/precond: `bf16`, `precond=frob`, `l_target=0.05`
- Trials: `20`, timing reps per trial: `5`

Raw logs are in `benchmark_results/2026_02_25/solve_ablation_t20/`.

## Winner
- **Default winner: `greedy-newton`**
- Vs `off`, `greedy-newton` mean iter delta: -5.93% (wins 17/18 cells)
- Vs `off`, `greedy-minimax` mean iter delta: -2.69% (wins 15/18 cells)

## Mode Deltas Vs `off` (Mean % Change)
| Mode | p=1 iter | p=2 iter | p=4 iter | overall iter | overall total | overall relerr |
|---|---:|---:|---:|---:|---:|---:|
| greedy-newton | -3.90% | -6.62% | -7.27% | -5.93% | -1.13% | -4.56% |
| greedy-minimax | -2.32% | +0.16% | -5.90% | -2.69% | +6.47% | -14.99% |
| auto (old mapping) | +1.04% | -2.17% | -6.63% | -2.59% | -2.32% | -0.35% |
| auto (new mapping) | +3.29% | -7.70% | -6.11% | -3.51% | +2.08% | +0.00% |

## Notes
- `greedy-minimax` improves p=1 relative error strongly, but is slower on average than `greedy-newton` in this suite.
- The final CLI default was switched to `--online-coeff-mode greedy-newton` for all `p` values.

