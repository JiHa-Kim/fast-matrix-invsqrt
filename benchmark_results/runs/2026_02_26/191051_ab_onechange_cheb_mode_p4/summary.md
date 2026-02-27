# A/B Summary: `p=4` Chebyshev Mode (`fixed` vs `minimax-auto`)

Date: 2026-02-26

Decision:
- Reject `--cheb-mode minimax-auto` as a default policy for maintained `p=4` benchmark runs.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024,SPD p=4 k=n=2048" --ab-extra-args-a="--cheb-mode fixed" --ab-extra-args-b="--cheb-mode minimax-auto" --ab-label-a cheb_fixed --ab-label-b cheb_minimax_auto --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p4/manifest.json`

Key results (`Chebyshev-Apply` rows only):
- Total across 20 cells: `+2.05%` total ms (`A=230.517 ms`, `B=235.240 ms`).
- `k<n` subset (12 cells): `+5.66%` (regression).
- `k=n` subset (8 cells): `+0.57%` (regression).
- Win/loss count: `9` faster, `11` slower.
- Relative error: unchanged in all compared cells.

Conclusion:
- `minimax-auto` is slower on aggregate for the `p=4` target matrix, so keep `fixed` as default.
