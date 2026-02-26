# Chebyshev Direct Apply Benchmark (`Z = A^{-1/2} B`)

*Updated from fresh runs on 2026-02-25.*

## Configuration

- Command family: `scripts/matrix_solve.py`
- `p=2`, `sizes=1024,2048`, `trials=3`, `timing_reps=5`
- `dtype=bf16`, `precond=frob`, `l_target=0.05`
- Cases: `gaussian_spd`, `illcond_1e6`
- Compared methods:
  - `PE-Quad-Inverse-Multiply`
  - `PE-Quad-Coupled-Apply`
  - `Chebyshev-Apply`

Raw logs:

- `artifacts/benchmarks/solve_p2_k16_2026-02-25.txt`
- `artifacts/benchmarks/solve_p2_k64_2026-02-25.txt`

## Results (`K=16`)

| Size | Case | Method | Iter ms | Peak Mem | RelErr vs Eig |
|---|---|---|---:|---:|---:|
| 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 8.429 | 28 MB | 3.906e-03 |
| 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 5.633 | 30 MB | 4.211e-03 |
| 1024 | gaussian_spd | Chebyshev-Apply | 5.680 | 21 MB | 3.174e-03 |
| 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 3.711 | 28 MB | 2.594e-03 |
| 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 2.344 | 30 MB | 4.974e-03 |
| 1024 | illcond_1e6 | Chebyshev-Apply | 10.549 | 21 MB | 3.052e-03 |
| 2048 | gaussian_spd | PE-Quad-Inverse-Multiply | 22.896 | 89 MB | 2.853e-03 |
| 2048 | gaussian_spd | PE-Quad-Coupled-Apply | 15.080 | 97 MB | 4.333e-03 |
| 2048 | gaussian_spd | Chebyshev-Apply | 11.328 | 57 MB | 3.235e-03 |
| 2048 | illcond_1e6 | PE-Quad-Inverse-Multiply | 23.366 | 89 MB | 2.838e-03 |
| 2048 | illcond_1e6 | PE-Quad-Coupled-Apply | 14.679 | 97 MB | 4.364e-03 |
| 2048 | illcond_1e6 | Chebyshev-Apply | 10.588 | 57 MB | 3.296e-03 |

## Results (`K=64`)

| Size | Case | Method | Iter ms | Peak Mem | RelErr vs Eig |
|---|---|---|---:|---:|---:|
| 1024 | gaussian_spd | PE-Quad-Inverse-Multiply | 8.946 | 29 MB | 3.891e-03 |
| 1024 | gaussian_spd | PE-Quad-Coupled-Apply | 3.572 | 31 MB | 4.211e-03 |
| 1024 | gaussian_spd | Chebyshev-Apply | 6.645 | 22 MB | 3.128e-03 |
| 1024 | illcond_1e6 | PE-Quad-Inverse-Multiply | 6.534 | 29 MB | 2.594e-03 |
| 1024 | illcond_1e6 | PE-Quad-Coupled-Apply | 7.597 | 31 MB | 4.974e-03 |
| 1024 | illcond_1e6 | Chebyshev-Apply | 17.965 | 22 MB | 3.036e-03 |
| 2048 | gaussian_spd | PE-Quad-Inverse-Multiply | 23.052 | 90 MB | 1.190e-03 |
| 2048 | gaussian_spd | PE-Quad-Coupled-Apply | 14.848 | 98 MB | 3.540e-03 |
| 2048 | gaussian_spd | Chebyshev-Apply | 8.595 | 59 MB | 3.113e-03 |
| 2048 | illcond_1e6 | PE-Quad-Inverse-Multiply | 22.788 | 90 MB | 1.289e-03 |
| 2048 | illcond_1e6 | PE-Quad-Coupled-Apply | 15.758 | 98 MB | 3.616e-03 |
| 2048 | illcond_1e6 | Chebyshev-Apply | 3.309 | 59 MB | 3.143e-03 |

## Takeaways

1. At `n=2048`, `Chebyshev-Apply` is consistently fastest in this sweep:
   - `1.33x` to `4.76x` faster than `PE-Quad-Coupled-Apply`.
   - `2.02x` to `6.89x` faster than `PE-Quad-Inverse-Multiply`.
2. At `n=1024`, performance is mixed and case-dependent; coupled apply often wins latency.
3. Chebyshev uses the least memory in every measured cell.
4. Relative error remains low across methods (roughly `1e-3` to `5e-3`).
