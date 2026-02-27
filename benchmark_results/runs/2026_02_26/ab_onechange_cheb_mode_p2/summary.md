# A/B Summary: `p=2` Chebyshev Mode (`fixed` vs `minimax-auto`)

Date: 2026-02-26

Decision:
- Reject `--cheb-mode minimax-auto` as a default policy for maintained `p=2` benchmark runs.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --ab-extra-args-a="--cheb-mode fixed" --ab-extra-args-b="--cheb-mode minimax-auto" --ab-label-a cheb_fixed --ab-label-b cheb_minimax_auto --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_mode_p2/manifest.json`

Key results (`Chebyshev-Apply` rows only):
- Total across 20 cells: `-0.08%` total ms (`A=226.935 ms`, `B=226.752 ms`).
- `k<n` subset (12 cells): `+1.57%` (regression).
- `k=n` subset (8 cells): `-0.75%`.
- Win/loss count: `10` faster, `10` slower.
- Relative error: unchanged in all compared cells.

Conclusion:
- No robust win for the target `p=2` matrix (`k<n` + `k=n`), so keep `fixed` as default.
