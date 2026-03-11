# fast-matrix-inverse-roots

Lean experimental kernels for two related problems:

- `polar/`: two-step rational polar iteration using product-form Zolotarev steps and Cholesky-only updates
- `fast_iroot/`: action-only baseline for computing G P^(-1/p) using Principled Gawlik minimax iterations

## Layout

- `polar_schedule.py`: CLI for the polar benchmark/demo runner
- `polar/`: modular polar implementation
- `fast_iroot/main.py`: unified inverse p-th root action baseline (p=2, 4)
- `ideas/polar/`: short notes on the current polar design

## Polar usage

```powershell
python polar_schedule.py --device cuda --mode demo --m 8192 --n 2048 --kappa_G 1e7 --schedule dwh3
```

The current fixed schedules are:

- `dwh3` as a baseline
- `dwh3_stable_solve` as the finite-precision-first rational path
- `dwh_tuned_fp32` as the fully 32-bit rational path

Exact success is measured with final `eigvalsh` verification.

## Inverse root usage

```powershell
python -m fast_iroot.main --p_root 4 --mode demo --m 2048 --n 256
```
