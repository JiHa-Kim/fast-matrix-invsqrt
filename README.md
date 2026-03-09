# fast-matrix-inverse-roots

Lean experimental kernels for two related problems:

- `polar/`: two-step rational polar iteration using product-form Zolotarev steps and Cholesky-only updates
- `fast_iroot/`: inverse fourth-root action baseline

## Layout

- `polar_schedule.py`: CLI for the polar benchmark/demo runner
- `polar/`: modular polar implementation
- `fast_iroot/inv_fourthroot.py`: inverse fourth-root runner and oracle checks
- `ideas/polar/`: short notes on the current polar design

## Polar usage

```powershell
python polar_schedule.py --device cuda --mode demo --m 8192 --n 2048 --kappa_G 1e7 --schedule zolo22
```

The current fixed schedules are:

- `zolo22` for the fast `1 + 2^-7` target
- `zolo32` for a stronger two-step fallback
- `dwh3` as a baseline

Exact success is measured with final `eigvalsh` verification.

## Inverse fourth-root usage

```powershell
python -m fast_iroot.inv_fourthroot --help
```
