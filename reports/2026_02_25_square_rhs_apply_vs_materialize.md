# Idea 3 Validation: Direct Coupled Apply vs Materialize-Then-Multiply (Square RHS)

*Date: 2026-02-25*

## Setup
- Script: `scripts/matrix_solve.py`
- RHS shape: square (`k=1024`)
- Exponents: `p in {1,2,4}`
- Cases: `gaussian_spd`, `illcond_1e6`
- Trials: `20`, timing reps: `5`
- Online coeff mode: `greedy-newton`

## Results
| p | case | inv-mul iter (ms) | coupled iter (ms) | iter delta | inv-mul total (ms) | coupled total (ms) | total delta | relerr inv-mul | relerr coupled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | gaussian_spd | 3.167 | 2.240 | -29.3% | 5.088 | 4.161 | -18.2% | 8.667e-03 | 8.240e-03 |
| 1 | illcond_1e6 | 3.160 | 2.244 | -29.0% | 4.690 | 3.774 | -19.5% | 5.814e-03 | 5.264e-03 |
| 2 | gaussian_spd | 3.927 | 2.550 | -35.1% | 5.441 | 4.064 | -25.3% | 2.441e-03 | 4.837e-03 |
| 2 | illcond_1e6 | 3.639 | 2.697 | -25.9% | 5.038 | 4.096 | -18.7% | 3.006e-03 | 5.280e-03 |
| 4 | gaussian_spd | 4.003 | 3.099 | -22.6% | 5.765 | 4.860 | -15.7% | 4.135e-03 | 3.998e-03 |
| 4 | illcond_1e6 | 4.260 | 3.282 | -23.0% | 6.849 | 5.872 | -14.3% | 3.517e-03 | 4.059e-03 |

- Mean iter delta (coupled vs inverse-multiply): **-27.5%**
- Mean total delta (coupled vs inverse-multiply): **-18.6%**
- Observation: even with square RHS, direct coupled apply remains faster by removing the final `X @ B` step.
