# A/B Summary: `p=2` Chebyshev CUDA Graph in Default Path

Date: 2026-02-26

Decision:
- Keep `--cheb-cuda-graph` enabled by default, even when global `--cuda-graph` is not set.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --ab-extra-args-a=--no-cheb-cuda-graph --ab-extra-args-b=--cheb-cuda-graph --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_graph_defaultpath_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_graph_defaultpath_p2/manifest.json`

Key results (`Chebyshev-Apply` rows only):
- Total across 20 cells: `-12.98%` total ms.
- Iter time aggregate: `-18.14%`.
- `k<n` subset (12 cells): `-36.72%`.
- `k=n` subset (8 cells): `-2.96%`.
- Win/loss count: `17` faster, `3` slower.
- Relative error: unchanged in all compared cells.
