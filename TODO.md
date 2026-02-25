# Project TODO

## Immediate Action Items (Bug Fixes & Hardening)
Refer to [BUGS.md](file:///d:/GitHub/JiHa-Kim/fast-matrix-inverse-roots/BUGS.md) for details.
- [x] Harden $p=2$ scalar scheduler.
- [x] Implement exact positivity and interval certification for $p=2$.
- [x] Add fallback to inverse Newton if certification fails.

## Roadmap

### Phase 1: flagship $p=2$ Scalar Scheduler
- [x] Add local-basis parameterization around $y=1$.
- [x] Implement exact positivity certification.
- [x] Replace sampled interval update with critical-point extrema.
- [x] Separate `true_interval` from `fit_interval`.

### Phase 2: $p=2$ Matrix Runtime
- [ ] Implement `fast` mode (storing $X$ and $Y$).
- [ ] Implement `low_mem` mode (storing only $X$).
- [ ] Add residual checks ($|I-Y|_F$).
- [ ] Benchmark against inverse Newton.

### Phase 3: Generic $p$ Support
- [ ] Generalize scalar objective and polynomial parameterization.
- [ ] Split odd/even $p$ branches.
- [ ] Add parity-specific positivity rules.
- [ ] Implement conservative interval certification (sampling + padding).

### Phase 4: Advanced Features
- [ ] Specialized $p=1$ and $p=4$ branches.
- [ ] Support for higher-degree polynomial families.
- [ ] Online schedule selection.
- [ ] Mixed precision support (bf16/fp16/fp32/fp64).