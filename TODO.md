# TODO

Date: 2026-02-26
Scope: prioritize unimplemented ideas from `ideas/p.md`, `ideas/p1.md`, and `ideas/1.md`, with emphasis on `p=2,4`.

## P0: Highest Priority (p=2,4)

- [ ] Implement staged updates for `p=2,4`: interval-focused minimax in early steps, then residual-binomial updates near `Y ~= I`.
  - Gap today: coupled PE runs fixed quadratic-family steps (plus affine/local alternatives), but no late binomial switch.
  - Exit check: beat current `Inverse-Newton-Coupled-Apply` or `PE-Quad-Coupled-Apply` on speed/relerr tradeoff for SPD `p=2,4` (`k<n` matrix).

- [ ] Implement full interval-minimax coefficient lookup by `(p, degree, kappa-bin)` for the contraction objective `max | t*q(t)^p - 1 |`.
  - Gap today: local minimax-alpha candidate exists, but not full degree-`d` runtime lookup from precomputed tables.
  - Exit check: online schedule uses nonzero minimax steps in real runs and improves total ms or relerr at fixed ms.

- [ ] Implement `p=4` two-stage strategy using tuned `p=2` kernels (instead of only direct `p=4` coupled updates).
  - Gap today: only direct `p=4` PE updates are wired.
  - Exit check: new method wins at least one important `p=4` regime (`n=1024/2048`, `k in {1,16,64}`).

- [ ] Add a precision-oriented correction pass for `p=2,4` apply paths.
  - Candidate: one cheap residual correction using the same approximate inverse-root apply operator.
  - Exit check: lower relerr vs reference with <20% extra iter time on target cells.

- [ ] Wire `Chebyshev-Apply` into the maintained `p=2,4` benchmark method list and auto-policy candidates.
  - Gap today: implementation exists, but default benchmark matrix does not include it for `p>1`.
  - Exit check: method appears in `run_benchmarks.py` default outputs and is compared in docs reports.

- [ ] Add an explicit `n=1024` policy benchmark track with `k in {16,64,1024}` and strategy recommendations.
  - Gap today: results exist, but there is no codified policy table for when to prefer Chebyshev vs coupled PE by `(n,k)`.
  - Exit check: benchmark report includes a decision table and a default strategy recommendation per `(n,k)` regime.

## P1: Next Wave

- [ ] Add stronger Turbo-style normalization for SPD: robust `lambda_min` estimation + scalar scaling target beyond current row-sum/Gershgorin proxies.
  - Gap today: `lambda_max` estimation exists; `lambda_min`-aware optimal scalar initialization is missing.

- [ ] Add block-diagonal SPD preconditioner mode in `precond_spd`.
  - Gap today: only none/frob/aol/jacobi/ruiz (diagonal/global scaling family) are available.

- [ ] Implement block-Lanczos direct apply for `A^{-1/p}B` (`p=2,4`) as an alternative to Chebyshev.
  - Gap today: no Lanczos/Krylov inverse-root apply kernel is present.

- [ ] Expose and benchmark `Torch-EVD-Solve` for `p>1` in the maintained benchmark matrix.
  - Gap today: evaluation code exists, but it is not in the default methods list.

- [ ] Add a mixed-precision factorization + iterative-refinement path for SPD `p>1` when `k ~= n`.
  - Gap today: direct polynomial methods are available, but there is no explicit high-accuracy mixed-precision fallback path for fat-RHS regimes.

## P2: Benchmark/Reporting Hygiene

- [ ] Add a single report view for `p=2,4` that tracks both speed and precision gap to strong references (EVD or high-precision baseline).
- [ ] Record per-run online schedule composition (`newton/minimax/affine`) in markdown summaries to see whether advanced policies are actually active.
- [ ] Add ML-relevant quality metrics to benchmark summaries: preconditioned-spectrum clustering probes and directional-bias proxies.
