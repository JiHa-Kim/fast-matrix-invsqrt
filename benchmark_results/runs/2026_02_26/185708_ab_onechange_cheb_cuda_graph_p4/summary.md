# A/B Summary: `p=4` Chebyshev CUDA Graph Toggle

Date: 2026-02-26

Decision:
- Keep `--cheb-cuda-graph` enabled by default for `Chebyshev-Apply` when `--cuda-graph` is on.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024,SPD p=4 k=n=2048" --extra-args=--cuda-graph --ab-extra-args-a=--no-cheb-cuda-graph --ab-extra-args-b=--cheb-cuda-graph --ab-label-a cheb_graph_off --ab-label-b cheb_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cheb_cuda_graph_p4/manifest.json`

Key results (`Chebyshev-Apply` rows only):
- Total across 20 cells: `-16.35%` total ms (`A=233.297 ms`, `B=195.144 ms`).
- `k<n` subset (12 cells): `-40.22%`.
- `k=n` subset (8 cells): `-6.36%`.
- Win/loss count: `17` faster, `3` slower.
- Relative error: unchanged in all compared cells.

Notable regressions (all small):
- `n=1024, k=1024, gaussian_spd`: `+5.09%`.
- `n=2048, k=2048, illcond_1e6`: `+1.93%`.
- `n=2048, k=2048, gaussian_spd`: `+0.45%`.
