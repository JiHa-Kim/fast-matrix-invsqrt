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
- [x] Implement `coupled` mode (storing $X$ and $Y$).
- [x] Implement `uncoupled` mode (storing only $X$).
- [x] Add residual checks ($|I-Y|_F$).
- [x] Benchmark against inverse Newton.

### Phase 3: Generic $p$ Support
- [x] Generalize scalar objective and polynomial parameterization.
- [x] Split odd/even $p$ branches.
- [x] Add parity-specific positivity rules.
- [x] Implement conservative interval certification (sampling + padding).

### Phase 4: Advanced Features
- [ ] Specialized $p=1$ and $p=4$ branches (for p=1, also implement an algorithm for direct matmat/matvec products solving $AX=B$ beyond just $A^{-1}$ then $A^{-1}B$ for numerical stability)
- [ ] Support for higher-degree polynomial families.
- [ ] Online schedule selection.
- [ ] Mixed precision support (bf16/fp16/fp32/fp64).