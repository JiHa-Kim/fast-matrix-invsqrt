# Known Bugs and Technical Debt

The following issues were identified in the current Polar-Express-inspired prototype and require immediate attention to ensure correctness and robustness.

- [x] **Unsafe Interval Propagation**: Current prototype uses a "cushion" (`l_cushion`) that clips the lower bound during propagation, which is unsafe for certification.
    - *Fix*: Keep `lo_true` and `hi_true` for propagation. Use `lo_fit = max(lo_true, l_cushion)` only for the fitting process.
- [x] **Lack of Positivity Certification**: The prototype relies on sampled penalties instead of exact certification of polynomial positivity.
    - *Fix*: Implement exact positivity checks for affine and quadratic polynomials on the target interval.
- [x] **Heuristic Interval Updates**: Prototyped interval updates use grid sampling, which can miss extrema.
    - *Fix*: Use critical-point extrema (derivative roots) for certified interval updates, especially for the flagship $p=2$ branch.
- [x] **Suboptimal Polynomial Parameterization**: Current parameterization is not centered around the fixed point.
    - *Fix*: Re-parameterize polynomials around $y=1$ and impose $q(1)=1$ and preferably $q'(1)=-1/p$ to improve conditioning and local convergence.
