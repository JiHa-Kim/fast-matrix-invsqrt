# A/B Summary: Local Binomial Candidate in `greedy-affine-opt`

Date: 2026-02-26  
Change under test (single change): add local-binomial quadratic candidate in `plan_coupled_quadratic_affine_opt_schedule` for late-stage `p in {2,4}`.

- `A` baseline commit: `b61ee37`
- `B` new working tree: binomial-candidate patch only

Benchmark scope (same settings both sides):
- `SPD p=2` matrix (includes `k<n` and `k=n` suites)
- `SPD p=4` matrix (includes `k<n` and `k=n` suites)
- `trials=5`, `timing_reps=5`, `timing_warmup_reps=2`, `dtype=bf16`

## Result

No clear robust win from this change.

- `p=2`: `PE-Quad-Coupled-Apply` total improved modestly (`-3.02%` over all 20 cells).
- `p=4`: `PE-Quad-Coupled-Apply` regressed (`+1.89%` over all 20 cells), with slight relerr drift (`rel_ratio ~= 1.002`).
- Fastest-method cell counts did not change (`14` PE-coupled, `6` Chebyshev for both A and B).

Because this single change does not improve both target exponents (`p=2,4`) under the same benchmark protocol, it should not be kept as-is.

## Decision

Reject this patch in current form and move to the next idea.
