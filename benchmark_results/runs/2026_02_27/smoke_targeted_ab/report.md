# Solver Benchmark A/B Report

Generated: 2026-02-26T20:39:59

A: baseline_cached
B: candidate

| kind | p | n | k | case | baseline_cached_method | candidate_method | baseline_cached_total_ms | candidate_total_ms | delta_ms(B-A) | delta_pct | baseline_cached_iter_ms | candidate_iter_ms | baseline_cached_relerr | candidate_relerr | relerr_ratio(B/A) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spd | 2 | 128 | 1 | gaussian_spd | Inverse-Newton-Coupled-Apply | PE-Quad-Coupled-Apply | 3769.732 | 3008.397 | -761.335 | -20.20% | 3675.461 | 2925.468 | 3.876e-03 | 4.120e-03 | 1.063 |
