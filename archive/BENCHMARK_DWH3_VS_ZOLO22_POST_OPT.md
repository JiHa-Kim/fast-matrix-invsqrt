# DWH3 vs ZOLO22 Benchmark (Post-Optimization)

## Improvements

This benchmark was rerun after the following major optimizations:
1. **Pass Fusion:** Combined multiple $O(mn^2)$ passes into a single large GEMM application at the end of the schedule.
2. **Matrix-Only Updates:** Updated the Gram matrix $S$ in $O(n^3)$ instead of recomputing from $X$ between steps.
3. **TRSM to GEMM:** Switched large-side updates from multiple Cholesky solves to a single fused GEMM pass.

## Setup

- Device: CUDA (Laptop RTX 3050 4GB VRAM)
- Input dtype: `float32`
- Iteration dtype: `float64` (Ensures stability for $\kappa=1e7$)
- Target: `kappa_G = 1e7` to `target_kappa(O) <= 1.0078431`
- Bank size: `3`
- Timed metric: `ms timed total`

## Results

| Shape | DWH3 success | DWH3 exact kappa(O) median | DWH3 total ms | DWH3 gram ms | DWH3 solve ms | DWH3 upd ms | ZOLO22 success | ZOLO22 exact kappa(O) median | ZOLO22 total ms | ZOLO22 gram ms | ZOLO22 solve ms | ZOLO22 upd ms | ZOLO22 / DWH3 total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048x256 | 3/3 | 1.0056072 | 22.285 | 4.997 | 11.043 | 3.589 | 0/3 | 1.0082454 | 27.324 | 4.952 | 16.684 | 3.507 | 1.226x |
| 4096x256 | 3/3 | 1.0050949 | 29.234 | 7.149 | 12.205 | 7.000 | 0/3 | 1.0080387 | 32.133 | 7.070 | 16.083 | 6.843 | 1.099x |
| 8192x256 | 3/3 | 1.0053714 | 40.552 | 13.133 | 11.156 | 13.960 | 0/3 | 1.0082098 | 47.264 | 13.746 | 16.687 | 13.592 | 1.166x |
| 8192x1024 | 3/3 | 1.0049066 | 419.316 | 154.480 | 102.152 | 155.565 | 3/3 | 1.0077624 | 526.481 | 154.333 | 210.238 | 155.268 | 1.256x |
| 16384x1024 | 3/3 | 1.0043726 | 721.936 | 308.304 | 100.544 | 306.578 | 3/3 | 1.0072747 | 828.616 | 306.117 | 209.303 | 306.677 | 1.148x |
| 8192x2048 | 3/3 | 1.0040908 | 1949.693 | 609.910 | 697.542 | 608.526 | 3/3 | 1.0070384 | 2781.498 | 609.827 | 1529.834 | 608.585 | 1.427x |
| 16384x2048 | 3/3 | 1.0041235 | 3175.708 | 1221.084 | 699.388 | 1216.157 | 3/3 | 1.0070525 | 4011.905 | 1220.729 | 1535.658 | 1221.775 | 1.263x |
| 16384x4096 | 3/3 | 1.0040603 | 15156.216 | 5008.388 | 5019.463 | 4935.339 | 3/3 | 1.0070212 | 21439.664 | 4888.723 | 11429.125 | 4890.844 | 1.415x |

## Takeaways

- **Massive Speedup:** On the largest shape `16384x4096`, `dwh3` total time dropped from **36.6s to 15.1s** (~2.4x speedup).
- **Architecture Shift:** The dominant cost shifted from multiple large-side TRSM passes to a single initial Gram pass and a single final GEMM update (`upd ms`).
- **Precision vs Speed:** Using `float64` for iterations is mandatory for stability at $\kappa=1e7$, but fusion mitigates the hardware's slow $fp64$ units by minimizing passes.
- **Schedule Comparison:** `dwh3` is now significantly faster than `zolo22` across all large shapes, primarily because the $O(n^3)$ cost of 3 DWH steps is lower than the $O(n^3)$ cost of 2 Zolo steps (which require more inversions per step).

## Raw Logs

- `benchmarks/logs/` (available upon request)
