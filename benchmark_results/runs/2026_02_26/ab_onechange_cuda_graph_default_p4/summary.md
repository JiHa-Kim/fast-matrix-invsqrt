# A/B Summary: `p=4` Global CUDA Graph Toggle (default-off vs on)

Date: 2026-02-26

Decision:
- Reject global `--cuda-graph` default-on.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=4 k<n,SPD p=4 k=n=256,SPD p=4 k=n=512,SPD p=4 k=n=1024,SPD p=4 k=n=2048" --ab-extra-args-b=--cuda-graph --ab-label-a cuda_graph_off --ab-label-b cuda_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p4/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p4/manifest.json`

Key results:
- All methods aggregate (100 rows): `-1.12%` total ms.
- `k<n`: `-4.41%`.
- `k=n`: `+4.09%` (regression).
- Relerr: unchanged for all compared rows.

Conclusion:
- Because `k=n` regresses on the `p=4` matrix, keep global `--cuda-graph` opt-in.
