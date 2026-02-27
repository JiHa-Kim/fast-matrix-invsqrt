# A/B Summary: `p=2` Chebyshev CUDA Graph Toggle

Date: 2026-02-26

Decision:
- Keep `--cheb-cuda-graph` enabled by default for `Chebyshev-Apply` when `--cuda-graph` is on.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --extra-args=--cuda-graph --ab-extra-args-a=--no-cheb-cuda-graph --ab-extra-args-b=--cheb-cuda-graph --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p2/manifest.json`

Key results (`Chebyshev-Apply` rows only):
- Total across 20 cells: `-13.39%` total ms (`A=226.387 ms`, `B=196.077 ms`).
- `k<n` subset (12 cells): `-31.16%`.
- `k=n` subset (8 cells): `-6.37%`.
- Win/loss count: `19` faster, `1` slower.
- Relative error: unchanged in all compared cells.

Notable worst-case regression:
- `n=2048, k=1, illcond_1e6`: `+0.77%` total ms.
