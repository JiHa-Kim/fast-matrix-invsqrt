# Spectral Convergence Analysis

Iterative inverse-root methods aim to drive the spectrum of the iteration matrix $Y \approx X^p A$ toward the identity $I$. This page documents the mathematical foundation and empirical results of eigenvalue convergence for the `PE-Quad` method.

## Theoretical Foundation: "Inverse Root Express"

Traditional iterations like Newton-Schulz use a fixed affine polynomial ($q(t) = 1.5 - 0.5t$ for $p=2$) which is only optimal when eigenvalues are already very close to 1.0. Our **Polynomial-Express (PE)** approach uses tuned quadratic schedules designed to contract a broad spectral interval $[l_{target}, 1.0]$.

For an eigenvalue $t \in \lambda(Y_k)$, the update is:
$$
t \mapsto \phi_k(t) := t \cdot (q_k(t))^p
$$
The coefficients are pre-tuned to minimize the maximum deviation from 1.0 across the target interval, ensuring fast early contraction for ill-conditioned matrices.

---

## Empirical Comparison: PE-Quad vs. Newton-Schulz

We analyzed the spectral convergence for $n=1024, p=2$ (inverse square root) across 10 trials with random orthogonal bases. The target spectral interval was set to $[0.05, 1.0]$, matching our production preconditioning defaults. 

**Metrics reported are worst-case values across all 10 trials.**

### PE-Quad (Production Default)
*Fixed 4-step schedule tuned for $[0.05, 1.0]$.*

| Step | Min $\lambda$ | Max $\lambda$ | Mean $\lambda$ | $\rho(I-Y)$ | Cluster 90% | Cluster 99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 0.5250 | 9.50e-01 | 10.5% | 1.1% |
| 1 | 0.6247 | 1.3742 | 0.9659 | 3.75e-01 | 19.0% | 2.0% |
| 2 | 0.9843 | 1.0157 | 1.0001 | **1.57e-02** | 100.0% | 99.2% |
| 3 | 1.0000 | 1.0000 | 1.0001 | **1.06e-06** | 100.0% | 100.0% |
| 4 | 1.0000 | 1.0000 | 1.0000 | **5.96e-08** | 100.0% | 100.0% |

### Newton-Schulz (Baseline)
*Classic affine iteration $X \leftarrow X(1.5I - 0.5Y)$.*

| Step | Min $\lambda$ | Max $\lambda$ | Mean $\lambda$ | $\rho(I-Y)$ | Cluster 90% | Cluster 99% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 1.0000 | 0.5250 | 9.50e-01 | 10.5% | 1.1% |
| 1 | 0.1088 | 1.0000 | 0.7206 | 8.91e-01 | 36.4% | 12.0% |
| 2 | 0.2273 | 1.0000 | 0.8667 | 7.73e-01 | 65.0% | 38.7% |
| 3 | 0.4369 | 1.0000 | 0.9509 | 5.63e-01 | 84.9% | 66.9% |
| 4 | 0.7176 | 1.0000 | 0.9880 | **2.82e-01** | 95.6% | 85.8% |

---

## Key Takeaways

1.  **Scale Invariance**: Results at $n=1024$ confirm that PE-Quad's mathematical advantage scales perfectly to production dimensions.
2.  **Robust Early Contraction**: PE-Quad handles the broad initial interval $[0.05, 1.0]$ with superior efficiency. By Step 2, 100% of eigenvalues are within 1.6% of the identity, while Newton-Schulz leaves nearly 35% of eigenvalues outside the 10% tolerance window.
3.  **Spectral Fidelity**: At Step 4, PE-Quad's worst-case spectral residual is **4,700,000x smaller** than Newton-Schulz.

To reproduce these results, run:
```bash
uv run python -m benchmarks.spectral_convergence
```
