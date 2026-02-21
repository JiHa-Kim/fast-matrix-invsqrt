# PE2 Method (Quadratic Polynomial Express)

Benchmark name:

- `PE2`

Core implementation:

- `isqrt_core.py:inverse_sqrt_pe_quadratic`

## Multiplier

Uses tuned quadratic coefficients:

$$
B_t = a_t I + b_t Y + c_t Y^2
$$

In current benchmark harness, `PE2` uses first 2 steps of a tuned quadratic schedule.

## Update Equations

Per full step:

$$
Y_2 \leftarrow Y Y
$$

$$
B_t = a_t I + b_t Y + c_t Y_2
$$

$$
X \leftarrow X B_t,\qquad
Y \leftarrow B_t Y B_t
$$

Optional:

$$
Y \leftarrow \frac{1}{2}(Y + Y^\top)
$$

Final step (terminal mode, default):

Still computes $Y_2$ and $B_t$, then:

$$
X \leftarrow X B_t
$$

Skip final $Y$ update.

## Mathematical Trick

Quadratic `q_t(Y)` gives stronger per-step shaping than affine methods, often reducing residual faster for hard spectra.

## Code Tricks Used

- explicit buffered `Y2` in workspace
- `matmul(out=...)` throughout
- pre-extracted CPU coefficient triples
- terminal last-step skip

## When It Usually Works Best

- difficult / spiky spectra where affine methods miss target residual
- larger sizes where stronger per-step contraction can offset extra work

## Tradeoffs

- extra compute to form `Y^2`
- can be overkill in easy cases
- coefficient tuning/safety materially affects robustness
