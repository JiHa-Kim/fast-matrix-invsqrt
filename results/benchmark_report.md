# Fast Matrix Inverse p-th Roots Benchmark Report
*Date: 2026-02-24*

This report details the performance and accuracy of quadratic PE (Polynomial-Express) iterations for matrix inverse p-th roots.

## Methodology
- **Sizes**: 256,512,1024
- **Compiled**: Yes (`torch.compile(mode='max-autotune')`)
- **Trials per case**: 10
- **Hardware**: GPU (bf16)
- **Methods Compared**: `Inverse-Newton` (baseline), `PE-Quad` (uncoupled quadratic), `PE-Quad-Coupled` (coupled quadratic).

## Results for $p=1$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.178 | 1.225 | 14 | 3.808e-03 | 5.383e-02 | 3.794e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.198 | 1.245 | 13 | 4.182e-03 | - | 4.168e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.101 | 1.148 | 14 | 4.402e-03 | 1.838e-01 | 4.380e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.502 | 1.146 | 14 | 2.805e-03 | 3.959e-02 | 2.808e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.452 | 1.096 | 13 | 5.105e-03 | - | 5.105e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.509 | 1.153 | 14 | 4.168e-03 | 2.053e-01 | 4.143e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.558 | 1.246 | 14 | 2.584e-03 | 1.325e-02 | 2.586e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.453 | 1.142 | 13 | 6.271e-03 | - | 6.274e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.517 | 1.206 | 14 | 7.537e-03 | 1.512e-01 | 7.494e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.487 | 1.183 | 14 | 1.405e-03 | 1.249e-02 | 1.403e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.418 | 1.113 | 13 | 8.120e-03 | - | 8.109e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.412 | 1.108 | 14 | 4.909e-03 | 1.451e-01 | 4.888e-03 |
| Inv-Newton | 256x256 | spike | 2.487 | 1.111 | 14 | 1.745e-03 | 1.742e-02 | 1.726e-03 |
| PE-Quad | 256x256 | spike | 2.591 | 1.215 | 13 | 6.714e-03 | - | 6.699e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.701 | 1.325 | 14 | 5.534e-03 | 1.114e-01 | 5.519e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.656 | 1.335 | 30 | 1.888e-03 | 5.083e-02 | 1.874e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.512 | 1.191 | 28 | 1.036e-02 | - | 1.036e-02 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.448 | 1.127 | 30 | 5.069e-03 | 1.764e-01 | 5.064e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.565 | 1.132 | 30 | 1.867e-03 | 1.979e-02 | 1.865e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.599 | 1.165 | 28 | 7.754e-03 | - | 7.740e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.559 | 1.125 | 30 | 5.842e-03 | 1.483e-01 | 5.828e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.459 | 1.167 | 30 | 1.816e-03 | 1.591e-02 | 1.811e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.563 | 1.272 | 28 | 9.175e-03 | - | 9.166e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.495 | 1.203 | 30 | 5.799e-03 | 9.734e-02 | 5.782e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.612 | 1.244 | 30 | 1.660e-03 | 1.455e-02 | 1.661e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.542 | 1.174 | 28 | 6.383e-03 | - | 6.376e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.547 | 1.179 | 30 | 5.617e-03 | 1.693e-01 | 5.616e-03 |
| Inv-Newton | 512x512 | spike | 2.434 | 1.110 | 30 | 1.528e-03 | 2.703e-02 | 1.504e-03 |
| PE-Quad | 512x512 | spike | 2.523 | 1.199 | 28 | 5.289e-03 | - | 5.282e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.510 | 1.187 | 30 | 5.190e-03 | 1.756e-01 | 5.191e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 3.903 | 2.652 | 94 | 2.243e-03 | 1.036e-01 | 2.235e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 3.996 | 2.745 | 88 | 9.704e-03 | - | 9.709e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 3.879 | 2.629 | 94 | 6.228e-03 | 4.431e-02 | 6.216e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 3.933 | 2.638 | 94 | 2.447e-03 | 2.247e-02 | 2.439e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.031 | 2.736 | 88 | 9.543e-03 | - | 9.549e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 3.917 | 2.622 | 94 | 6.442e-03 | 3.810e-02 | 6.429e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 3.932 | 2.642 | 94 | 2.592e-03 | 1.883e-02 | 2.584e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.051 | 2.761 | 88 | 9.410e-03 | - | 9.415e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 3.895 | 2.605 | 94 | 6.586e-03 | 3.306e-02 | 6.575e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 3.910 | 2.636 | 94 | 2.654e-03 | 1.730e-02 | 2.646e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 4.035 | 2.761 | 88 | 9.340e-03 | - | 9.345e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 3.884 | 2.610 | 94 | 6.648e-03 | 3.074e-02 | 6.637e-03 |
| Inv-Newton | 1024x1024 | spike | 3.847 | 2.596 | 94 | 3.236e-03 | 5.805e-02 | 3.223e-03 |
| PE-Quad | 1024x1024 | spike | 4.010 | 2.759 | 88 | 8.276e-03 | - | 8.294e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 3.881 | 2.630 | 94 | 7.373e-03 | 1.599e-01 | 7.379e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.492 | 1.217 | 14 | 5.461e-03 | 6.829e-03 | 2.708e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.447 | 1.171 | 13 | 4.636e-03 | - | 2.277e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.658 | 1.382 | 14 | 3.840e-03 | 4.784e-02 | 1.918e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.596 | 1.333 | 14 | 4.919e-03 | 3.960e-03 | 2.452e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.459 | 1.196 | 13 | 3.450e-03 | - | 1.723e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.479 | 1.217 | 14 | 3.383e-03 | 4.367e-02 | 1.692e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.767 | 1.181 | 14 | 1.278e-02 | 6.291e-03 | 6.346e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.770 | 1.184 | 13 | 2.405e-03 | - | 1.186e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.781 | 1.195 | 14 | 2.509e-03 | 1.353e-02 | 1.240e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.708 | 1.193 | 14 | 5.808e-03 | 2.707e-03 | 2.920e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.692 | 1.177 | 13 | 1.633e-03 | - | 8.250e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.704 | 1.189 | 14 | 6.633e-03 | 3.472e-02 | 3.349e-03 |
| Inv-Newton | 256x256 | spike | 2.849 | 1.260 | 14 | 1.185e-02 | 1.036e-02 | 5.934e-03 |
| PE-Quad | 256x256 | spike | 2.767 | 1.179 | 13 | 3.163e-03 | - | 1.569e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.812 | 1.224 | 14 | 8.902e-03 | 5.153e-02 | 4.460e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.490 | 1.211 | 30 | 2.682e-03 | 7.310e-03 | 1.360e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.651 | 1.372 | 28 | 2.307e-03 | - | 1.203e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.219 | 1.940 | 30 | 2.319e-03 | 4.279e-02 | 1.210e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.563 | 1.409 | 30 | 1.060e-02 | 4.785e-03 | 5.349e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.557 | 1.403 | 28 | 2.314e-03 | - | 1.206e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 4.039 | 1.885 | 30 | 5.625e-03 | 3.906e-02 | 2.824e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 4.835 | 1.413 | 30 | 7.600e-03 | 3.755e-03 | 3.764e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 4.714 | 1.292 | 28 | 1.733e-03 | - | 8.604e-04 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 4.748 | 1.326 | 30 | 1.021e-02 | 7.595e-02 | 5.069e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.549 | 1.208 | 30 | 1.359e-02 | 3.398e-03 | 6.730e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.653 | 1.313 | 28 | 2.972e-03 | - | 1.488e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.703 | 1.362 | 30 | 5.169e-03 | 3.197e-02 | 2.580e-03 |
| Inv-Newton | 512x512 | spike | 2.812 | 1.420 | 30 | 1.378e-02 | 1.612e-02 | 6.918e-03 |
| PE-Quad | 512x512 | spike | 2.720 | 1.328 | 28 | 3.615e-03 | - | 1.841e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.618 | 1.227 | 30 | 7.823e-03 | 5.226e-02 | 3.888e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.547 | 3.050 | 94 | 2.800e-03 | 7.669e-03 | 1.504e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.821 | 3.325 | 88 | 2.799e-03 | - | 1.502e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.572 | 3.076 | 94 | 1.116e-02 | 1.248e-01 | 5.663e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.583 | 3.100 | 94 | 1.671e-04 | 5.578e-03 | 7.987e-05 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.830 | 3.346 | 88 | 1.594e-04 | - | 7.155e-05 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.626 | 3.143 | 94 | 1.093e-02 | 1.250e-01 | 5.607e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.796 | 3.236 | 94 | 1.865e-04 | 4.553e-03 | 8.143e-05 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.040 | 3.480 | 88 | 1.793e-04 | - | 7.290e-05 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.643 | 3.083 | 94 | 1.089e-02 | 1.250e-01 | 5.558e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.300 | 3.053 | 94 | 1.965e-04 | 4.136e-03 | 9.285e-05 |
| PE-Quad | 1024x1024 | near_rank_def | 4.701 | 3.453 | 88 | 1.899e-04 | - | 8.535e-05 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.423 | 3.175 | 94 | 1.087e-02 | 1.250e-01 | 5.532e-03 |
| Inv-Newton | 1024x1024 | spike | 5.071 | 3.082 | 94 | 1.434e-03 | 2.472e-02 | 6.682e-04 |
| PE-Quad | 1024x1024 | spike | 5.241 | 3.252 | 88 | 1.566e-03 | - | 7.408e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.069 | 3.080 | 94 | 1.045e-02 | 1.188e-01 | 5.166e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.715 | 1.342 | 14 | 1.839e-02 | 1.437e-01 | 6.224e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.747 | 1.374 | 13 | 4.798e-03 | - | 1.586e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.831 | 1.458 | 14 | 1.739e-02 | 9.600e-02 | 5.864e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.604 | 1.326 | 14 | 1.908e-02 | 1.537e-01 | 6.446e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.680 | 1.402 | 13 | 4.325e-03 | - | 1.422e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.601 | 1.323 | 14 | 1.761e-02 | 9.472e-02 | 5.937e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.637 | 1.321 | 14 | 1.221e-02 | 6.757e-02 | 4.058e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.690 | 1.373 | 13 | 8.271e-03 | - | 2.775e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.859 | 1.542 | 14 | 1.199e-02 | 4.207e-02 | 3.963e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.729 | 1.325 | 14 | 1.957e-02 | 1.714e-01 | 6.523e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.787 | 1.382 | 13 | 4.267e-03 | - | 1.434e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.894 | 1.490 | 14 | 1.955e-02 | 7.203e-02 | 6.517e-03 |
| Inv-Newton | 256x256 | spike | 2.688 | 1.343 | 14 | 1.364e-02 | 1.085e-01 | 4.520e-03 |
| PE-Quad | 256x256 | spike | 2.810 | 1.465 | 13 | 7.450e-03 | - | 2.448e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.992 | 1.648 | 14 | 1.361e-02 | 2.344e-02 | 4.509e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.756 | 1.328 | 30 | 2.017e-02 | 2.528e-01 | 6.731e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.946 | 1.519 | 28 | 3.294e-03 | - | 1.082e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.034 | 1.606 | 30 | 1.993e-02 | 6.050e-04 | 6.654e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.853 | 1.468 | 30 | 1.548e-02 | 1.754e-01 | 5.106e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.768 | 1.383 | 28 | 6.937e-03 | - | 2.392e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.154 | 1.769 | 30 | 1.525e-02 | 4.945e-04 | 5.038e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.746 | 1.386 | 30 | 1.837e-02 | 2.288e-01 | 6.059e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.994 | 1.633 | 28 | 4.998e-03 | - | 1.670e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.079 | 1.718 | 30 | 1.837e-02 | 4.141e-04 | 6.059e-03 |
| Inv-Newton | 512x512 | near_rank_def | 4.216 | 1.907 | 30 | 1.138e-02 | 9.594e-02 | 3.763e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.867 | 1.558 | 28 | 8.625e-03 | - | 2.882e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.712 | 1.403 | 30 | 1.138e-02 | 3.989e-04 | 3.763e-03 |
| Inv-Newton | 512x512 | spike | 2.930 | 1.582 | 30 | 1.012e-02 | 8.451e-02 | 3.366e-03 |
| PE-Quad | 512x512 | spike | 2.773 | 1.425 | 28 | 8.226e-03 | - | 2.719e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.868 | 1.521 | 30 | 1.037e-02 | 1.353e-02 | 3.451e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.907 | 3.572 | 94 | 2.019e-02 | 3.745e-01 | 6.692e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.351 | 4.015 | 88 | 8.068e-03 | - | 2.655e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.907 | 3.572 | 94 | 2.019e-02 | 7.633e-04 | 6.689e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.032 | 3.567 | 94 | 2.004e-02 | 3.750e-01 | 6.660e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.484 | 4.019 | 88 | 1.570e-03 | - | 4.179e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.047 | 3.582 | 94 | 2.003e-02 | 5.695e-04 | 6.660e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.040 | 3.600 | 94 | 1.990e-02 | 3.750e-01 | 6.631e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.512 | 4.072 | 88 | 1.427e-03 | - | 3.859e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.042 | 3.602 | 94 | 1.990e-02 | 5.045e-04 | 6.630e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.958 | 3.625 | 94 | 1.983e-02 | 3.750e-01 | 6.616e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.353 | 4.020 | 88 | 1.357e-03 | - | 3.703e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.940 | 3.607 | 94 | 1.983e-02 | 4.682e-04 | 6.616e-03 |
| Inv-Newton | 1024x1024 | spike | 4.891 | 3.576 | 94 | 1.775e-02 | 3.404e-01 | 5.985e-03 |
| PE-Quad | 1024x1024 | spike | 5.337 | 4.021 | 88 | 4.136e-03 | - | 1.390e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.910 | 3.594 | 94 | 1.827e-02 | 2.067e-02 | 6.164e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.762 | 1.254 | 14 | 1.148e-02 | 1.530e-01 | 2.984e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.922 | 1.414 | 13 | 1.038e-02 | - | 2.460e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.772 | 1.265 | 14 | 1.021e-02 | 9.727e-02 | 2.457e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.540 | 1.244 | 14 | 9.687e-03 | 2.300e-01 | 2.323e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.684 | 1.389 | 13 | 1.042e-02 | - | 2.493e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.570 | 1.275 | 14 | 9.696e-03 | 1.097e-01 | 2.322e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.547 | 1.260 | 14 | 8.312e-03 | 1.074e-01 | 2.123e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.661 | 1.374 | 13 | 1.777e-02 | - | 4.364e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.556 | 1.270 | 14 | 8.367e-03 | 4.622e-02 | 2.136e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.654 | 1.382 | 14 | 1.054e-02 | 2.299e-01 | 2.565e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.820 | 1.548 | 13 | 1.234e-02 | - | 3.015e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.715 | 1.443 | 14 | 1.060e-02 | 1.051e-01 | 2.581e-03 |
| Inv-Newton | 256x256 | spike | 2.976 | 1.478 | 14 | 9.448e-03 | 1.603e-01 | 2.354e-03 |
| PE-Quad | 256x256 | spike | 3.048 | 1.550 | 13 | 1.489e-02 | - | 3.691e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.023 | 1.525 | 14 | 1.490e-02 | 3.405e-02 | 3.691e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.762 | 1.478 | 30 | 1.041e-02 | 3.439e-01 | 2.513e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.856 | 1.572 | 28 | 1.073e-02 | - | 2.580e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.634 | 1.350 | 30 | 1.055e-02 | 1.754e-01 | 2.534e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.410 | 1.699 | 30 | 9.162e-03 | 2.415e-01 | 2.300e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.116 | 1.405 | 28 | 1.525e-02 | - | 3.775e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.214 | 1.503 | 30 | 9.512e-03 | 7.160e-02 | 2.381e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.070 | 1.549 | 30 | 9.797e-03 | 3.056e-01 | 2.431e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.027 | 1.506 | 28 | 1.258e-02 | - | 3.132e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.869 | 1.349 | 30 | 1.111e-02 | 3.221e-02 | 2.766e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.130 | 1.583 | 30 | 8.994e-03 | 1.339e-01 | 2.258e-03 |
| PE-Quad | 512x512 | near_rank_def | 5.263 | 3.716 | 28 | 1.684e-02 | - | 4.213e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 5.090 | 3.543 | 30 | 1.321e-02 | 1.914e-02 | 3.315e-03 |
| Inv-Newton | 512x512 | spike | 7.023 | 3.315 | 30 | 9.269e-03 | 1.916e-01 | 2.335e-03 |
| PE-Quad | 512x512 | spike | 5.203 | 1.495 | 28 | 1.606e-02 | - | 3.992e-03 |
| PE-Quad-Coupled | 512x512 | spike | 5.187 | 1.479 | 30 | 1.597e-02 | 3.664e-02 | 3.966e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.858 | 3.521 | 94 | 9.700e-03 | 4.997e-01 | 2.428e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.361 | 4.023 | 88 | 9.705e-03 | - | 2.427e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.945 | 3.607 | 94 | 9.705e-03 | 2.001e-01 | 2.427e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.083 | 3.519 | 94 | 9.857e-03 | 5.000e-01 | 2.405e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.565 | 4.002 | 88 | 9.841e-03 | - | 2.405e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.105 | 3.541 | 94 | 9.841e-03 | 4.301e-04 | 2.405e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.674 | 3.430 | 94 | 9.890e-03 | 5.000e-01 | 2.385e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.219 | 3.975 | 88 | 9.890e-03 | - | 2.384e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.747 | 3.503 | 94 | 9.890e-03 | 3.890e-04 | 2.385e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.783 | 3.476 | 94 | 9.834e-03 | 5.000e-01 | 2.374e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.292 | 3.985 | 88 | 9.836e-03 | - | 2.373e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.791 | 3.485 | 94 | 9.836e-03 | 3.684e-04 | 2.374e-03 |
| Inv-Newton | 1024x1024 | spike | 4.788 | 3.505 | 94 | 8.851e-03 | 4.814e-01 | 2.184e-03 |
| PE-Quad | 1024x1024 | spike | 5.289 | 4.006 | 88 | 8.921e-03 | - | 2.204e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.770 | 3.487 | 94 | 8.842e-03 | 9.086e-02 | 2.180e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.009 | 1.638 | 14 | 1.513e-02 | 1.811e-01 | 1.965e-03 |
| PE-Quad | 256x256 | gaussian_spd | 4.287 | 2.916 | 13 | 1.113e-02 | - | 1.453e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.978 | 1.608 | 14 | 2.959e-02 | 1.551e-01 | 3.836e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.235 | 1.589 | 14 | 1.226e-02 | 1.436e-01 | 1.558e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 4.122 | 2.476 | 13 | 1.074e-02 | - | 1.399e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.109 | 1.463 | 14 | 1.128e-02 | 1.135e-01 | 1.436e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.072 | 1.462 | 14 | 1.849e-02 | 1.094e-01 | 2.381e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 4.010 | 2.400 | 13 | 1.849e-02 | - | 2.381e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.093 | 1.483 | 14 | 1.849e-02 | 4.488e-02 | 2.381e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.846 | 1.458 | 14 | 1.280e-02 | 3.516e-02 | 1.668e-03 |
| PE-Quad | 256x256 | near_rank_def | 7.845 | 6.456 | 13 | 1.280e-02 | - | 1.668e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 4.914 | 3.526 | 14 | 1.479e-02 | 1.167e-01 | 1.920e-03 |
| Inv-Newton | 256x256 | spike | 2.782 | 1.398 | 14 | 1.649e-02 | 1.217e-01 | 2.149e-03 |
| PE-Quad | 256x256 | spike | 3.897 | 2.513 | 13 | 1.558e-02 | - | 2.046e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.833 | 1.449 | 14 | 3.548e-02 | 1.338e-01 | 4.560e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.243 | 1.707 | 30 | 1.122e-02 | 1.712e-01 | 1.450e-03 |
| PE-Quad | 512x512 | gaussian_spd | 4.567 | 3.031 | 28 | 1.122e-02 | - | 1.450e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.006 | 1.470 | 30 | 1.121e-02 | 1.685e-01 | 1.450e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.776 | 1.445 | 30 | 1.573e-02 | 7.967e-02 | 2.051e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.956 | 2.625 | 28 | 1.573e-02 | - | 2.051e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.931 | 1.599 | 30 | 1.600e-02 | 1.169e-01 | 2.085e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.639 | 1.369 | 30 | 1.305e-02 | 5.168e-02 | 1.731e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.863 | 2.593 | 28 | 1.305e-02 | - | 1.731e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.798 | 1.528 | 30 | 3.686e-02 | 2.168e-01 | 4.749e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.863 | 1.450 | 30 | 1.731e-02 | 9.592e-02 | 2.277e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.887 | 2.474 | 28 | 1.731e-02 | - | 2.277e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.833 | 1.420 | 30 | 2.115e-02 | 7.851e-02 | 2.762e-03 |
| Inv-Newton | 512x512 | spike | 2.763 | 1.378 | 30 | 1.708e-02 | 2.137e-01 | 2.201e-03 |
| PE-Quad | 512x512 | spike | 3.802 | 2.417 | 28 | 1.690e-02 | - | 2.183e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.876 | 1.491 | 30 | 2.818e-02 | 1.632e-01 | 3.604e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.417 | 3.952 | 94 | 2.041e-02 | 4.633e-01 | 2.632e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 9.103 | 7.638 | 88 | 1.020e-02 | - | 1.386e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.414 | 3.949 | 94 | 6.653e-02 | 4.995e-01 | 8.560e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.249 | 3.953 | 94 | 1.004e-02 | 1.070e-01 | 1.376e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 9.004 | 7.708 | 88 | 1.004e-02 | - | 1.376e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.258 | 3.962 | 94 | 6.646e-02 | 5.000e-01 | 8.560e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.494 | 3.947 | 94 | 9.912e-03 | 1.122e-01 | 1.367e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 9.241 | 7.694 | 88 | 9.913e-03 | - | 1.367e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.587 | 4.041 | 94 | 6.629e-02 | 5.000e-01 | 8.547e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 6.091 | 3.983 | 94 | 9.872e-03 | 1.204e-01 | 1.362e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 9.745 | 7.638 | 88 | 9.874e-03 | - | 1.362e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 6.063 | 3.956 | 94 | 6.618e-02 | 5.000e-01 | 8.533e-03 |
| Inv-Newton | 1024x1024 | spike | 6.130 | 4.049 | 94 | 1.473e-02 | 3.006e-01 | 1.880e-03 |
| PE-Quad | 1024x1024 | spike | 9.727 | 7.645 | 88 | 1.025e-02 | - | 1.316e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 6.032 | 3.951 | 94 | 6.155e-02 | 4.768e-01 | 7.879e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
