# A/B Summary: `p=2` Global CUDA Graph Toggle (default-off vs on)

Date: 2026-02-26

Decision:
- Do not switch global `--cuda-graph` to default-on based on this run set.

Command:
- `uv run python benchmarks/run_benchmarks.py --only "SPD p=2 k<n,SPD p=2 k=n=256,SPD p=2 k=n=512,SPD p=2 k=n=1024,SPD p=2 k=n=2048" --ab-extra-args-b=--cuda-graph --ab-label-a cuda_graph_off --ab-label-b cuda_graph_on --ab-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p2/report.md --manifest-out benchmark_results/runs/2026_02_26/ab_onechange_cuda_graph_default_p2/manifest.json`

Key results:
- All methods aggregate (100 rows): `-5.08%` total ms.
- `k<n`: `-3.73%`.
- `k=n`: `-7.00%`.
- Relerr: unchanged for all compared rows.

Interpretation:
- `p=2` looks favorable in aggregate, but default policy decision is tied to both `p=2` and `p=4`.
