# Mathematical Foundations for GEMM-Robustness in bf16

When deploying iterative preconditioning algorithms (like computing the inverse square root) in pure machine learning environments, we face strict performance constraints:
1. **Pure BF16 Loop**: To maximize hardware efficiency and avoid memory bandwidth bottlenecks, we execute the entire cycle—including symmetric accumulation and matrix multiplications (GEMMs)—exclusively in `bf16`.
2. **Quantization Noise**: `bf16` has only 7 bits of mantissa. The relative machine epsilon is $u \approx 3.9 \times 10^{-3}$, which introduces substantial non-deterministic error during large GEMMs.

This document formalizes the mathematics of the robustness guards needed to maintain stability in a pure `bf16` environment.

---

## 1. The Scaling Factor $\beta$ and the Overshoot Catastrophe

At each iteration, we form the preconditioned matrix:
$$S = Z^T B Z$$
To apply a polynomial $q(x)$ designed to be stable on the interval $[\ell, 1]$, we must scale $S$ so that its maximum eigenvalue does not exceed $1$.

We compute an estimate $\beta \ge \lambda_{\max}(S)$ and scale:
$$\widehat{S} = \frac{S}{\beta}$$

### The Danger
If our estimate $\beta$ is slightly too small due to finite-precision reduction errors, or if the actual computed $\lambda_{\max}(S)$ in hardware drifts above the theoretical $\lambda_{\max}$, we get $\lambda_{\max}(\widehat{S}) = 1 + \delta$.
For Chebyshev polynomials evaluated via Clenshaw recurrence, evaluating outside the $[-1, 1]$ domain (which maps to our $x \in [\ell, 1]$ domain) causes **exponential divergence**. The polynomial shoots to infinity, destroying the matrix $Z$.

---

## 2. Padding $\beta$: Absolute vs. Relative

To protect against $\lambda_{\max}(\widehat{S}) > 1$, we pad the estimate:
- **Absolute Padding**: $\beta \leftarrow \beta + \epsilon_{\text{abs}}$
- **Relative Padding**: $\beta \leftarrow \beta(1 + \epsilon_{\text{rel}})$

**Which is mathematically correct for BF16 workloads?**

The error in computing $S = Z^T B Z$ via `bf16` GEMMs satisfies standard backward error bounds. For matrices of dimension $n$, the error $E$ in the product satisfies:
$$ \|E\|_2 \le \gamma_n u \|Z\|_2^2 \|B\|_2 $$
where $u \approx 3.9 \times 10^{-3}$ for `bf16`, and $\gamma_n = \frac{nu}{1-nu}$.

Because $S_{\text{exact}} = Z^T B Z$, we know $\|S_{\text{exact}}\|_2 \le \|Z\|_2^2 \|B\|_2$.
Therefore, the error bound scales **multiplicatively** with the magnitude of $S$:
$$ \|E\|_2 \lesssim \gamma_n u \|S\|_2 $$
Since $\beta \approx \|S\|_2$, the uncertainty in the maximum eigenvalue of the computed $S$ is directly proportional to $\beta$.

### Conclusion on Padding
If we use absolute padding ($\beta + \epsilon_{\text{abs}}$):
- When $\beta$ is very large (early iterations), the fixed $\epsilon_{\text{abs}}$ is dwarfed by the multiplicative GEMM error, failing to prevent overshoot and causing divergence.
- When $\beta$ is very small (late iterations, as $S \to I$, $\beta \to 1$), the fixed $\epsilon_{\text{abs}}$ unnecessarily over-damps the matrix, slowing convergence.

**Relative padding ($\beta \leftarrow \beta(1 + \epsilon_{\text{rel}})$) is strictly superior.** It perfectly matches the geometry of floating-point arithmetic.
For `bf16`, a relative padding of $\epsilon_{\text{rel}} \approx 10^{-2}$ safely absorbs the GEMM noise, guaranteeing $\lambda_{\max}(S/\beta) \le 1$.

---

## 3. Mandatory Symmetrization in Pure bf16

The polynomial recurrences (Horner and Clenshaw) inherently assume that the input matrix $\widehat{S}$ is symmetric. If $\widehat{S}$ is non-symmetric, it possesses complex eigenvalues. High-degree polynomials act chaotically on complex eigenvalues, breaking the preconditioner.

In a pure `bf16` workflow, we achieve sufficient robustness by explicitly enforcing symmetry at the lowest cost possible:
$$ S \leftarrow \frac{1}{2}(S + S^T) $$

This operation is memory-bound but computationally trivial. By enforcing exact structural symmetry in `bf16`, we guarantee that the spectrum remains strictly real, honoring the assumptions of the minimax polynomial design.

---

## 4. Unified Scaling Policy: Enforcing Parity

To compute $\beta$, we have three standard estimators based on matrix norms/traces:
1. **Frobenius Norm**: $\beta_{\text{fro}} = \|S\|_F$
2. **Trace**: $\beta_{\text{trace}} = \frac{1}{n} \text{Tr}(S)$
3. **Max Diagonal**: $\beta_{\text{maxdiag}} = \max_i S_{ii}$

Because $S$ is symmetric positive definite (SPD):
$$ \max_i S_{ii} \le \lambda_{\max}(S) \le \|S\|_F \le \text{Tr}(S) $$

Using $\|S\|_F$ is a safe, rigorous upper bound for $\lambda_{\max}$, but it requires computing the full sum of squares. As $S \to I$, $\|S\|_F \to \sqrt{n}$, meaning the $\beta$ value diverges relative to the true $\lambda_{\max} \to 1$.

To enforce parity across these estimators so they seamlessly act as drop-in replacements for one another without altering the fundamental dynamics of the loop, we normalize them to bound $\lambda_{\max} \approx 1$:
- For Frobenius: $\beta_{\text{fro\_normalized}} = \frac{1}{\sqrt{n}} \|S\|_F$
- For Trace: $\beta_{\text{trace}} = \frac{1}{n} \text{Tr}(S)$

By enforcing that every estimator returns exactly $1.0$ when $S = I$, we decouple the polynomial scaling logic from the matrix dimension $n$.

### The Final $\beta$ Protocol
1. Calculate raw estimator $e$ (e.g., $\frac{1}{\sqrt{n}}\|S\|_F$).
2. Clamp below to ensure non-collapse: $e = \max(e, 1.0)$.
3. Apply relative safety padding: $\beta = e \times (1 + \epsilon_\beta)$.
4. Scale: $\widehat{S} = S / \beta$.
