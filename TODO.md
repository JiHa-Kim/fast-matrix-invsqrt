# TODO

## Working Agreement
- Run one-change A/B experiments where possible.
- Default-policy changes require a strict win:
  - better or comparable quality/stability (`relerr`, `relerr_p90`, `fail_rate`), and
  - clear runtime win, and
  - better assessment score aggregate.
- If not strictly better, reject/archive and document in `docs/methods/benchmark_decisions.md`.

## Current
- [x] Decision-grade A/B for non-SPD `p=1` renorm policy (`renorm_every=0` vs `1`).
  - Run: `benchmark_results/runs/2026_02_27/225808_ab_nonspd_p1_renorm_step13/`
  - Outcome: rejected as default (`renorm_on` slower in all matched cells; mixed quality; better score in 1/12).

## Next
- [ ] A/B `nonspd_safe_early_metric`: `diag` vs `fro` on maintained non-SPD `p=1 k<n`.
- [ ] Add per-case assessment-score deltas to decision summaries.
- [ ] Continue algorithm iteration on adaptive coefficient policies with fairness-controlled A/B.

## Completed
- [x] Benchmark assessment overhaul (`relerr_p90`, `fail_rate`, `q_per_ms`, assessment score).
- [x] A/B summary and assessment leaders in benchmark markdown outputs.
- [x] A/B interleaving support (`--ab-interleave`) for fairer comparisons.
- [x] Coupled renorm knobs + safer non-SPD early gate metric wiring.
- [x] Regression fix: `fro` early gate works when adaptive mode is disabled.
