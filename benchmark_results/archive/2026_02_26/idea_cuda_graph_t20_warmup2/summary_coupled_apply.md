# CUDA Graph Ablation (20 Trials, Warmup=2)

- Setup: `matrix_solve.py`, `n=1024`, `p in {1,2,4}`, `k in {1,16,64}`, cases `{gaussian_spd, illcond_1e6}`, `precond=jacobi`, `bf16`, `trials=20`, `timing_reps=5`, `timing_warmup_reps=2`.
- Method: `PE-Quad-Coupled-Apply` with `--online-coeff-mode greedy-affine-opt`.
- Compared: `--cuda-graph off` vs `--cuda-graph on` (warmup=3).

## Overall Means (Coupled Apply)
| variant | count | total_ms | iter_ms | relerr | newton_steps | minimax_steps | affineopt_steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 18 | 3.622 | 2.011 | 4.198e-03 | 0.33 | 0.00 | 0.67 |
| on | 18 | 3.522 | 1.938 | 4.198e-03 | 0.33 | 0.00 | 0.67 |

- Delta (on vs off): total **-2.76%**, iter **-3.59%**, relerr **+0.00%**.
- Cell wins (lower total_ms): on **12**, off **6**, ties **0** (out of 18).

## Per-p Deltas (means over k x case)
| p | off_total_ms | on_total_ms | d_total | d_iter | d_relerr |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.053 | 3.029 | -0.79% | -5.12% | +0.00% |
| 2 | 3.573 | 3.464 | -3.05% | -3.25% | +0.00% |
| 4 | 4.241 | 4.074 | -3.94% | -2.96% | +0.00% |

## Per-k Deltas (means over p x case)
| k | off_total_ms | on_total_ms | d_total | d_iter | d_relerr |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.575 | 3.480 | -2.64% | -3.42% | +0.00% |
| 16 | 3.662 | 3.457 | -5.59% | -4.06% | +0.00% |
| 64 | 3.630 | 3.629 | -0.02% | -3.31% | +0.00% |

## Extremes (total_ms delta)
- Best cell: p=4, k=16, case=gaussian_spd, delta=-14.47% (4.553 -> 3.894 ms)
- Worst cell: p=1, k=64, case=gaussian_spd, delta=+5.72% (2.939 -> 3.107 ms)

- Raw parsed rows: `benchmark_results/2026_02_26/idea_cuda_graph_t20_warmup2/comparison_coupled_apply.csv`
