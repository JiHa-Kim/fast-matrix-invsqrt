# DWH3 vs ZOLO22 Benchmark

## Setup

- Device: CUDA
- Input / iteration dtype: `float32`
- Target: `kappa_G = 1e7` to `target_kappa(O) <= 1.0078431`
- Verification: exact final `eigvalsh` on CUDA
- Bank size: `3`
- Chunking: `gram_chunk_rows=2048`, `rhs_chunk_rows=2048`
- Timed metric: `ms timed total`

`ms exact_verify` is excluded from the timed total below.

## Commands

```powershell
python polar_schedule.py --device cuda --mode bank --m <m> --n <n> --kappa_G 1e7 --target_mode robust --schedule dwh3 --bank_size 3 --exact_verify_device cuda
python polar_schedule.py --device cuda --mode bank --m <m> --n <n> --kappa_G 1e7 --target_mode robust --schedule zolo22 --bank_size 3 --exact_verify_device cuda
```

## Results

| Shape | DWH3 success | DWH3 exact kappa(O) median | DWH3 total ms | DWH3 gram ms | DWH3 solve ms | ZOLO22 success | ZOLO22 exact kappa(O) median | ZOLO22 total ms | ZOLO22 gram ms | ZOLO22 solve ms | ZOLO22 / DWH3 total |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2048x256 | 3/3 | 1.0039641 | 45.616 | 14.116 | 29.512 | 3/3 | 1.0069775 | 52.268 | 10.852 | 39.327 | 1.146x |
| 4096x256 | 3/3 | 1.0039641 | 74.890 | 26.065 | 46.814 | 3/3 | 1.0069775 | 69.407 | 17.098 | 50.475 | 0.927x |
| 8192x256 | 3/3 | 1.0039638 | 124.258 | 44.571 | 76.720 | 3/3 | 1.0069775 | 130.347 | 33.636 | 94.700 | 1.049x |
| 8192x1024 | 3/3 | 1.0039645 | 1232.445 | 610.710 | 615.050 | 3/3 | 1.0069777 | 1284.615 | 458.087 | 820.097 | 1.042x |
| 16384x1024 | 3/3 | 1.0039645 | 2434.278 | 1218.069 | 1211.300 | 3/3 | 1.0069777 | 2534.911 | 917.645 | 1610.995 | 1.041x |
| 8192x2048 | 3/3 | 1.0039645 | 4794.555 | 2432.681 | 2328.493 | 3/3 | 1.0069777 | 4962.211 | 1825.670 | 3102.995 | 1.035x |
| 16384x2048 | 3/3 | 1.0039645 | 9483.089 | 4878.811 | 4570.263 | 3/3 | 1.0069777 | 9786.174 | 3661.235 | 6091.388 | 1.032x |
| 16384x4096 | 3/3 | 1.0039645 | 36669.670 | 19481.882 | 16957.478 | 3/3 | 1.0069778 | 37555.502 | 14696.626 | 22624.205 | 1.024x |

## Takeaways

- Both schedules hit the `1 + 2^-7` style target on every tested shape.
- `dwh3` is stronger in exact final conditioning: about `1.0039645` versus `1.0069778` for `zolo22`.
- `dwh3` is also faster on every tested shape except `4096x256`.
- As `n` grows, the main difference is in solve time:
  - `zolo22` saves one iteration, but each Zolo step is more expensive.
  - `dwh3` pays for one extra step, but the DWH solve/update is cheaper per step.
- On the largest tested shape, `16384x4096`, `dwh3` is about `2.4%` faster.

## Raw Logs

- `data/bench_dwh3_vs_zolo22_dwh3.txt`
- `data/bench_dwh3_vs_zolo22_zolo22.txt`
