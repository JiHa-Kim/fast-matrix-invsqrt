# Two-Step Rational Polar Iteration with Cholesky-Only Updates

## Main point

The working two-step GPU path is not the old partial-fraction realization

$$
U = \widehat m \left(I + \sum_j a_j (S + c_j I)^{-1}\right),
$$

but the product-form realization

$$
U = \widehat m \prod_j (S + c^{\mathrm{even}}_j I)(S + c^{\mathrm{odd}}_j I)^{-1}
  = \widehat m \prod_j \left(I + (c^{\mathrm{even}}_j - c^{\mathrm{odd}}_j)(S + c^{\mathrm{odd}}_j I)^{-1}\right),
$$

applied directly to the tall matrix in row chunks. This still uses only Cholesky
factorizations of shifted SPD systems, but it avoids the catastrophic
cancellation of the sum form on CUDA.

## Scalar contraction lemma

Let

$$
r(x) = x \phi(x^2), \qquad x \in [\ell,1], \qquad 0 < \ell \le 1,
$$

and define the polar iteration

$$
X_+ = X \phi(X^\top X).
$$

Assume:

1. $r(1) = 1$,
2. $r(x) > 0$ for $x \in [\ell,1]$,
3. $r$ is nondecreasing on $[\ell,1]$.

If the singular values of $X$ lie in $[\ell,1]$, then the singular values of
$X_+$ lie in $[r(\ell),1]$. Therefore

$$
\kappa_O(X_+) \le \frac{1}{r(\ell)}.
$$

For two steps $r_1, r_2$,

$$
\kappa_O(X_2) \le \frac{1}{r_2(r_1(\ell_0))}.
$$

### Proof

Take an SVD $X = U \Sigma V^\top$. Then

$$
X^\top X = V \Sigma^2 V^\top,
\qquad
\phi(X^\top X) = V \phi(\Sigma^2) V^\top.
$$

Hence

$$
X_+ = X \phi(X^\top X)
    = U \Sigma \phi(\Sigma^2) V^\top
    = U \operatorname{diag}(r(\sigma_i)) V^\top.
$$

So the new singular values are exactly $r(\sigma_i)$. Since
$\sigma_i \in [\ell,1]$ and $r$ is nondecreasing,

$$
r(\ell) \le r(\sigma_i) \le r(1) = 1.
$$

Thus $\sigma(X_+) \subset [r(\ell),1]$, giving

$$
\kappa_O(X_+) \le \frac{1}{r(\ell)}.
$$

Applying the same argument twice yields

$$
\kappa_O(X_2) \le \frac{1}{r_2(r_1(\ell_0))}.
$$

## Practical consequence

The analytic guarantee is entirely scalar. The matrix-side realization only needs
to be numerically faithful enough to track the scalar map. This is why the
product-form Cholesky realization matters: it preserves the same scalar map as
the exact Zolotarev step while avoiding the numerical instability of the
partial-fraction sum.

## Current fixed schedules

The repository now keeps a small fixed schedule set instead of a large search
surface:

1. `zolo22` for the fast target near $1 + 2^{-7}$,
2. `zolo32` as the stronger two-step fallback,
3. `dwh3` as the baseline.

All Zolo schedules use the product-form Cholesky realization. The scalar
guarantee is still

$$
\kappa_O^{(2)} \le \frac{1}{r_2(r_1(\ell_0))},
$$

but the repository surface is smaller and easier to benchmark cleanly.
