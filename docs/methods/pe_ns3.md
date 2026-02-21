# PE-NS3 Method (Affine Polynomial Express)

Benchmark name:

- `PE-NS3`

Core implementation:

- `isqrt_core.py:inverse_sqrt_pe_affine`

## Multiplier

Uses per-step tuned affine coefficients:

$$
B_t = a_t I + b_t Y
$$

with a 3-step schedule (`T=3`).

## Update Equations

Per full step:

$$
X \leftarrow X B_t,\qquad
Y \leftarrow B_t Y B_t
$$

Optional:

$$
Y \leftarrow \frac{1}{2}(Y + Y^\top)
$$

Final step (terminal mode, default):

$$
X \leftarrow X B_t
$$

Skip final $Y$ update.

## Mathematical Trick

Compared with NS, this keeps the same affine structure but lets each step use tuned `(a_t, b_t)` to better shape spectral contraction over the target interval.

## Code Tricks Used

- coefficients pre-extracted once to CPU float pairs
- no per-step GPU scalar extraction
- shared workspace and `matmul(out=...)`
- terminal final-step skip

## When It Usually Works Best

- when affine shaping beats fixed NS coefficients for given spectrum
- medium difficulty regimes where PE2 extra work is not needed

## Tradeoffs

- same structural cost class as NS, but not always faster in wall-clock
- sensitive to coefficient quality and floor target assumptions
