# A/B Summary: Add `Chebyshev-Apply` to `p>1` solve method matrix

Date: 2026-02-26  
Change under test (single change): commit `7d4aa3d` vs baseline `e82a4bb`

- `A` (baseline): no `Chebyshev-Apply` in `matrix_solve_methods(p>1)`
- `B` (new): include `Chebyshev-Apply` for `p>1`

Benchmark commands (identical for A/B per p):
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n" --trials 5 --timing-reps 5 --timing-warmup-reps 2 --dtype bf16`
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n" --trials 5 --timing-reps 5 --timing-warmup-reps 2 --dtype bf16`

## p=2 (SPD, k<n)

Totals across 12 cells (`n in {1024,2048}`, `k in {1,16,64}`, 2 cases):
- `A` best total (existing methods): `PE-Quad-Coupled-Apply = 87.711 ms`
- `B` new method: `Chebyshev-Apply = 65.181 ms`

Fastest-per-cell count:
- `A`: `PE-Quad-Coupled-Apply` wins `12/12`
- `B`: `Chebyshev-Apply` wins `6/12`, `PE-Quad-Coupled-Apply` wins `6/12`

Pattern:
- `n=1024`: Chebyshev is slower than best non-Cheb by about `+1.9` to `+2.5 ms`
- `n=2048`: Chebyshev is faster by about `-4.8` to `-6.1 ms`

## p=4 (SPD, k<n)

Totals across 12 cells:
- `A` best total (existing methods): `PE-Quad-Coupled-Apply = 102.705 ms`
- `B` new method: `Chebyshev-Apply = 65.153 ms`

Fastest-per-cell count:
- `A`: `PE-Quad-Coupled-Apply` wins `12/12`
- `B`: `Chebyshev-Apply` wins `6/12`, `PE-Quad-Coupled-Apply` wins `6/12`

Pattern:
- `n=1024`: Chebyshev is slower than best non-Cheb by about `+1.45` to `+2.02 ms`
- `n=2048`: Chebyshev is faster by about `-6.63` to `-7.79 ms`

## Conclusion

This single change is validated: adding `Chebyshev-Apply` to the maintained `p>1` method matrix exposes real wins on larger sizes (`n=2048`) while keeping existing methods available for smaller-size regimes (`n=1024`).
