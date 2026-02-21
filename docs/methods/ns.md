# NS Method (Newton-Schulz)

Applies to benchmark names:

- `NS3` (3 iterations)
- `NS4` (4 iterations)

Core implementation:

- `isqrt_core.py:inverse_sqrt_ns`

## Multiplier

For each step:

$$
B = 1.5I - 0.5Y
$$

This is the classic affine Newton-Schulz multiplier.

## Update Equations

Per full step:

$$
X \leftarrow X B,\qquad
Y \leftarrow B Y B
$$

Optional symmetrization:

$$
Y \leftarrow \frac{1}{2}(Y + Y^\top)
$$

Final step (terminal mode, default):

$$
X \leftarrow X B
$$

Skip final $Y$ update.

## Code Tricks Used

- workspace reuse for all temporaries
- `matmul(out=...)` to avoid allocations
- optional symmetrization to limit drift
- terminal last-step skip for lower GEMM count

## When It Usually Works Best

- moderate conditioning after preconditioning
- cases where simple affine dynamics already hit residual target
- often strong baseline in small-to-medium sizes

## Tradeoffs

- fixed affine form can converge slower on harder spectra versus PE2
- very stable and simple, but may need extra step on difficult cases
