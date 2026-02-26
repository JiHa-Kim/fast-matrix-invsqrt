# IRoot Preconditioner Ablation (PE-Quad-Coupled, 20 Trials)

- Parsed rows: 60 (expected 60)
- Cells: 15 (p in {1,2,4}, cases=5)

## Overall Means (lower is better)
| mode | count | total_ms | iter_ms | pre_ms | resid | d_total_vs_frob | d_iter_vs_frob | d_pre_vs_frob | d_resid_vs_frob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aol | 15 | 4.771 | 2.917 | 1.854 | 9.533e-03 | -0.54% | -1.97% | +1.79% | -8.89% |
| frob | 15 | 4.796 | 2.975 | 1.821 | 1.046e-02 | +0.00% | +0.00% | +0.00% | +0.00% |
| jacobi | 15 | 4.686 | 2.943 | 1.744 | 9.205e-03 | -2.29% | -1.10% | -4.25% | -12.02% |
| ruiz | 15 | 4.794 | 2.925 | 1.870 | 8.858e-03 | -0.05% | -1.70% | +2.66% | -15.34% |

## Per-p Means (total_ms)
| mode | p=1 | p=2 | p=4 |
|---|---:|---:|---:|
| aol | 4.558 | 4.592 | 5.163 |
| frob | 4.035 | 5.292 | 5.062 |
| jacobi | 4.158 | 4.707 | 5.194 |
| ruiz | 4.252 | 4.894 | 5.237 |

## Cell Wins (best total_ms per cell)
| mode | wins |
|---|---:|
| aol | 3 |
| frob | 6 |
| jacobi | 6 |
| ruiz | 0 |

