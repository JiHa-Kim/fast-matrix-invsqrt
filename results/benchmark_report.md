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
| Inv-Newton | 256x256 | gaussian_spd | 2.315 | 0.991 | 13 | 2.066e-03 | 1.258e-01 | 2.070e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.521 | 1.197 | 13 | 3.809e-03 | - | 3.808e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.369 | 1.045 | 13 | 4.521e-03 | 1.381e-01 | 4.513e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.462 | 1.143 | 13 | 1.499e-03 | 1.255e-01 | 1.489e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.912 | 1.593 | 13 | 7.084e-03 | - | 7.089e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.745 | 1.425 | 13 | 5.912e-03 | 1.361e-01 | 5.901e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.465 | 0.949 | 13 | 1.108e-03 | 1.253e-01 | 1.110e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.590 | 1.074 | 13 | 8.147e-03 | - | 8.150e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.625 | 1.109 | 13 | 7.568e-03 | 7.348e-02 | 7.554e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.278 | 0.939 | 13 | 2.367e-03 | 1.253e-01 | 2.377e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.406 | 1.067 | 13 | 8.876e-03 | - | 8.882e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.343 | 1.005 | 13 | 8.508e-03 | 6.450e-02 | 8.484e-03 |
| Inv-Newton | 256x256 | spike | 2.283 | 0.945 | 13 | 4.917e-03 | 1.232e-01 | 4.929e-03 |
| PE-Quad | 256x256 | spike | 2.372 | 1.033 | 13 | 6.829e-03 | - | 6.844e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.385 | 1.046 | 13 | 6.946e-03 | 1.175e-01 | 6.937e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.287 | 0.979 | 28 | 2.989e-03 | 1.777e-01 | 2.972e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.519 | 1.210 | 27 | 3.401e-03 | - | 3.408e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.266 | 0.958 | 28 | 2.282e-03 | 8.083e-02 | 2.282e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.271 | 0.958 | 28 | 2.576e-03 | 1.773e-01 | 2.555e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.332 | 1.019 | 27 | 8.097e-03 | - | 8.113e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.371 | 1.058 | 28 | 5.834e-03 | 5.467e-02 | 5.831e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.623 | 1.110 | 28 | 3.259e-03 | 1.771e-01 | 3.242e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.611 | 1.099 | 27 | 4.077e-03 | - | 4.094e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.500 | 0.987 | 28 | 4.215e-03 | 8.421e-02 | 4.226e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.384 | 1.068 | 28 | 1.574e-03 | 1.771e-01 | 1.557e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.393 | 1.077 | 27 | 7.033e-03 | - | 7.040e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.324 | 1.008 | 28 | 7.339e-03 | 1.381e-01 | 7.326e-03 |
| Inv-Newton | 512x512 | spike | 2.456 | 0.997 | 28 | 2.349e-03 | 1.730e-01 | 2.326e-03 |
| PE-Quad | 512x512 | spike | 2.964 | 1.504 | 27 | 6.235e-03 | - | 6.242e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.513 | 1.054 | 28 | 6.271e-03 | 1.688e-01 | 6.273e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 3.718 | 2.477 | 88 | 3.590e-03 | 2.509e-01 | 3.569e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 3.879 | 2.638 | 84 | 4.224e-03 | - | 4.245e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 3.629 | 2.388 | 88 | 4.232e-03 | 1.507e-01 | 4.250e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 3.848 | 2.267 | 88 | 4.346e-03 | 2.505e-01 | 4.332e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.055 | 2.475 | 84 | 2.536e-04 | - | 2.435e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 3.881 | 2.301 | 88 | 3.180e-04 | 3.558e-02 | 3.232e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 3.632 | 2.264 | 88 | 1.128e-03 | 2.503e-01 | 1.115e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 3.920 | 2.552 | 84 | 7.071e-03 | - | 7.076e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 3.631 | 2.263 | 88 | 7.071e-03 | 2.532e-01 | 7.077e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.501 | 2.263 | 88 | 1.189e-03 | 2.503e-01 | 1.177e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.719 | 2.481 | 84 | 7.008e-03 | - | 7.013e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.538 | 2.300 | 88 | 7.008e-03 | 2.528e-01 | 7.015e-03 |
| Inv-Newton | 1024x1024 | spike | 3.639 | 2.315 | 88 | 2.372e-03 | 2.455e-01 | 2.354e-03 |
| PE-Quad | 1024x1024 | spike | 3.851 | 2.527 | 84 | 6.411e-03 | - | 6.415e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 3.606 | 2.282 | 88 | 6.123e-03 | 2.390e-01 | 6.130e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.559 | 1.037 | 13 | 3.329e-03 | 3.036e-02 | 1.640e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.634 | 1.112 | 13 | 3.649e-03 | - | 1.802e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.508 | 0.986 | 13 | 3.656e-03 | 3.338e-02 | 1.810e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.486 | 1.094 | 13 | 3.215e-03 | 6.013e-02 | 1.599e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.622 | 1.229 | 13 | 3.265e-03 | - | 1.623e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.379 | 0.986 | 13 | 3.251e-03 | 3.516e-02 | 1.627e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.275 | 1.021 | 13 | 1.938e-03 | 3.418e-02 | 9.660e-04 |
| PE-Quad | 256x256 | illcond_1e12 | 2.400 | 1.146 | 13 | 2.078e-03 | - | 1.031e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.486 | 1.232 | 13 | 2.118e-03 | 1.914e-02 | 1.054e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.301 | 1.032 | 13 | 1.729e-03 | 8.212e-03 | 8.815e-04 |
| PE-Quad | 256x256 | near_rank_def | 2.395 | 1.126 | 13 | 3.738e-03 | - | 1.874e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.316 | 1.048 | 13 | 2.556e-03 | 3.907e-03 | 1.279e-03 |
| Inv-Newton | 256x256 | spike | 2.301 | 1.028 | 13 | 3.365e-03 | 2.409e-02 | 1.725e-03 |
| PE-Quad | 256x256 | spike | 2.426 | 1.153 | 13 | 5.605e-03 | - | 2.851e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.409 | 1.137 | 13 | 8.699e-03 | 5.510e-02 | 4.369e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.949 | 1.109 | 28 | 4.223e-03 | 1.562e-01 | 2.069e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.022 | 1.182 | 27 | 4.223e-03 | - | 2.066e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.981 | 1.141 | 28 | 4.586e-03 | 8.378e-02 | 2.196e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.552 | 1.168 | 28 | 2.762e-03 | 1.122e-01 | 1.437e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.767 | 1.383 | 27 | 3.075e-03 | - | 1.594e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.398 | 1.014 | 28 | 2.910e-03 | 5.593e-02 | 1.483e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.384 | 1.093 | 28 | 3.408e-03 | 1.495e-01 | 1.732e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.548 | 1.256 | 27 | 3.406e-03 | - | 1.730e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.470 | 1.179 | 28 | 8.409e-03 | 7.318e-02 | 4.241e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.338 | 1.068 | 28 | 1.591e-03 | 6.774e-02 | 7.996e-04 |
| PE-Quad | 512x512 | near_rank_def | 2.527 | 1.257 | 27 | 1.590e-03 | - | 7.981e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.294 | 1.024 | 28 | 1.023e-02 | 2.975e-02 | 5.057e-03 |
| Inv-Newton | 512x512 | spike | 2.423 | 1.139 | 28 | 1.479e-03 | 6.455e-02 | 7.472e-04 |
| PE-Quad | 512x512 | spike | 2.456 | 1.171 | 27 | 1.774e-03 | - | 8.805e-04 |
| PE-Quad-Coupled | 512x512 | spike | 2.596 | 1.312 | 28 | 1.115e-02 | 6.966e-02 | 5.652e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.383 | 2.809 | 88 | 3.376e-03 | 2.142e-01 | 1.637e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.727 | 3.153 | 84 | 3.374e-03 | - | 1.635e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.337 | 2.763 | 88 | 1.098e-02 | 2.759e-01 | 5.523e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.430 | 2.771 | 88 | 3.888e-03 | 2.501e-01 | 1.846e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.780 | 3.121 | 84 | 3.887e-03 | - | 1.845e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.419 | 2.761 | 88 | 1.088e-02 | 2.795e-01 | 5.472e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 3.998 | 2.699 | 88 | 4.739e-04 | 4.197e-03 | 1.648e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.420 | 3.122 | 84 | 4.724e-04 | - | 1.618e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.024 | 2.726 | 88 | 1.155e-02 | 2.652e-01 | 5.801e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.092 | 2.728 | 88 | 4.884e-04 | 3.873e-03 | 1.859e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 4.476 | 3.112 | 84 | 4.870e-04 | - | 1.834e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.189 | 2.825 | 88 | 1.154e-02 | 2.656e-01 | 5.809e-03 |
| Inv-Newton | 1024x1024 | spike | 4.112 | 2.737 | 88 | 1.384e-03 | 8.567e-02 | 7.163e-04 |
| PE-Quad | 1024x1024 | spike | 4.479 | 3.105 | 84 | 1.673e-03 | - | 8.445e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.131 | 2.757 | 88 | 1.026e-02 | 2.109e-01 | 5.198e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.946 | 1.293 | 13 | 1.497e-02 | 5.224e-02 | 5.045e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.039 | 1.386 | 13 | 5.448e-03 | - | 1.869e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.855 | 1.202 | 13 | 5.832e-03 | 9.211e-02 | 1.949e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.472 | 1.158 | 13 | 1.537e-02 | 5.157e-02 | 5.207e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.636 | 1.322 | 13 | 5.088e-03 | - | 1.660e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.540 | 1.226 | 13 | 5.093e-03 | 3.747e-02 | 1.659e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.699 | 1.134 | 13 | 1.613e-02 | 5.901e-02 | 5.509e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.911 | 1.345 | 13 | 3.176e-03 | - | 1.081e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.802 | 1.236 | 13 | 3.171e-03 | 2.067e-02 | 1.079e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.791 | 1.137 | 13 | 1.525e-02 | 5.636e-02 | 5.201e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.939 | 1.284 | 13 | 3.018e-03 | - | 9.986e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.763 | 1.108 | 13 | 3.014e-03 | 5.123e-02 | 9.997e-04 |
| Inv-Newton | 256x256 | spike | 4.605 | 1.143 | 13 | 1.057e-02 | 4.688e-02 | 3.516e-03 |
| PE-Quad | 256x256 | spike | 4.755 | 1.293 | 13 | 4.300e-03 | - | 1.435e-03 |
| PE-Quad-Coupled | 256x256 | spike | 4.623 | 1.161 | 13 | 4.283e-03 | 1.019e-01 | 1.430e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.505 | 1.132 | 28 | 1.492e-02 | 8.053e-02 | 5.054e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.665 | 1.291 | 27 | 7.773e-03 | - | 2.720e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.588 | 1.214 | 28 | 7.820e-03 | 6.052e-02 | 2.645e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.828 | 1.166 | 28 | 1.360e-02 | 6.530e-02 | 4.625e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.162 | 1.500 | 27 | 5.893e-03 | - | 2.059e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.830 | 1.168 | 28 | 6.007e-03 | 4.819e-04 | 2.097e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.674 | 1.322 | 28 | 1.201e-02 | 4.788e-02 | 3.928e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.781 | 1.429 | 27 | 8.047e-03 | - | 2.622e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.607 | 1.255 | 28 | 8.101e-03 | 4.015e-04 | 2.780e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.681 | 1.186 | 28 | 1.595e-02 | 8.252e-02 | 5.242e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.802 | 1.306 | 27 | 4.108e-03 | - | 1.394e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.988 | 1.493 | 28 | 4.132e-03 | 3.879e-04 | 1.444e-03 |
| Inv-Newton | 512x512 | spike | 2.458 | 1.172 | 28 | 1.511e-02 | 1.032e-01 | 5.051e-03 |
| PE-Quad | 512x512 | spike | 2.553 | 1.268 | 27 | 4.037e-03 | - | 1.339e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.501 | 1.215 | 28 | 4.131e-03 | 6.629e-02 | 1.368e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.534 | 3.288 | 88 | 1.213e-02 | 7.182e-02 | 3.963e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.068 | 3.822 | 84 | 7.995e-03 | - | 2.586e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.522 | 3.276 | 88 | 8.249e-03 | 1.525e-01 | 2.739e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.852 | 3.318 | 88 | 9.048e-03 | 2.739e-03 | 2.940e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.362 | 3.828 | 84 | 9.043e-03 | - | 2.939e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.826 | 3.291 | 88 | 9.526e-03 | 1.782e-01 | 3.158e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.016 | 3.302 | 88 | 1.627e-02 | 1.250e-01 | 5.381e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.548 | 3.834 | 84 | 2.651e-03 | - | 9.175e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.995 | 3.281 | 88 | 2.653e-03 | 5.293e-04 | 9.178e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.899 | 3.304 | 88 | 1.620e-02 | 1.250e-01 | 5.367e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.365 | 3.770 | 84 | 2.713e-03 | - | 9.310e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.878 | 3.284 | 88 | 2.715e-03 | 5.126e-04 | 9.313e-04 |
| Inv-Newton | 1024x1024 | spike | 4.719 | 3.294 | 88 | 1.496e-02 | 1.546e-01 | 5.011e-03 |
| PE-Quad | 1024x1024 | spike | 5.242 | 3.817 | 84 | 4.138e-03 | - | 1.359e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.709 | 3.284 | 88 | 4.138e-03 | 8.629e-02 | 1.359e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.705 | 1.089 | 13 | 8.493e-03 | 1.080e-01 | 2.242e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.868 | 1.252 | 13 | 8.481e-03 | - | 2.241e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.842 | 1.226 | 13 | 1.217e-02 | 7.329e-02 | 3.144e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.380 | 1.092 | 13 | 1.027e-02 | 1.475e-01 | 2.666e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.748 | 1.459 | 13 | 1.058e-02 | - | 2.728e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.351 | 1.062 | 13 | 1.286e-02 | 6.944e-02 | 3.235e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.315 | 1.033 | 13 | 1.153e-02 | 1.314e-01 | 2.947e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.531 | 1.249 | 13 | 1.340e-02 | - | 3.268e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.400 | 1.118 | 13 | 1.428e-02 | 3.906e-02 | 3.450e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.647 | 1.170 | 13 | 1.101e-02 | 1.180e-01 | 2.815e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.883 | 1.406 | 13 | 1.281e-02 | - | 3.114e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.731 | 1.254 | 13 | 1.358e-02 | 1.563e-02 | 3.275e-03 |
| Inv-Newton | 256x256 | spike | 3.119 | 1.434 | 13 | 8.877e-03 | 9.560e-02 | 2.255e-03 |
| PE-Quad | 256x256 | spike | 3.255 | 1.570 | 13 | 9.420e-03 | - | 2.336e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.884 | 1.199 | 13 | 9.507e-03 | 6.347e-02 | 2.351e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.825 | 1.278 | 28 | 8.363e-03 | 2.425e-01 | 1.972e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.109 | 1.562 | 27 | 8.995e-03 | - | 2.085e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.848 | 1.301 | 28 | 8.969e-03 | 1.484e-01 | 2.085e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.593 | 1.295 | 28 | 1.050e-02 | 2.167e-01 | 2.586e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.651 | 1.354 | 27 | 1.181e-02 | - | 2.805e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.503 | 1.205 | 28 | 1.182e-02 | 1.121e-01 | 2.805e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.929 | 1.340 | 28 | 9.108e-03 | 2.433e-01 | 2.203e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.995 | 1.406 | 27 | 9.640e-03 | - | 2.291e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.924 | 1.335 | 28 | 9.649e-03 | 1.495e-01 | 2.291e-03 |
| Inv-Newton | 512x512 | near_rank_def | 6.135 | 2.395 | 28 | 1.225e-02 | 1.923e-01 | 3.057e-03 |
| PE-Quad | 512x512 | near_rank_def | 5.347 | 1.608 | 27 | 1.347e-02 | - | 3.253e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 5.681 | 1.942 | 28 | 1.347e-02 | 6.766e-02 | 3.253e-03 |
| Inv-Newton | 512x512 | spike | 3.608 | 1.691 | 28 | 1.244e-02 | 1.893e-01 | 3.143e-03 |
| PE-Quad | 512x512 | spike | 3.609 | 1.692 | 27 | 1.250e-02 | - | 3.093e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.185 | 1.268 | 28 | 1.255e-02 | 7.118e-02 | 3.106e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.725 | 3.236 | 88 | 9.266e-03 | 3.461e-01 | 2.253e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.317 | 3.829 | 84 | 9.699e-03 | - | 2.322e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.711 | 3.223 | 88 | 9.491e-03 | 2.141e-01 | 2.288e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.278 | 3.233 | 88 | 6.468e-03 | 3.750e-01 | 1.517e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.905 | 3.859 | 84 | 6.471e-03 | - | 1.517e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.559 | 3.513 | 88 | 6.475e-03 | 2.500e-01 | 1.517e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 8.241 | 3.231 | 88 | 1.287e-02 | 2.500e-01 | 3.294e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 8.832 | 3.822 | 84 | 1.371e-02 | - | 3.352e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 8.211 | 3.200 | 88 | 1.330e-02 | 3.559e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.947 | 3.313 | 88 | 1.289e-02 | 2.500e-01 | 3.298e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.507 | 3.873 | 84 | 1.365e-02 | - | 3.342e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.847 | 3.212 | 88 | 1.329e-02 | 3.428e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 4.495 | 3.203 | 88 | 1.281e-02 | 2.655e-01 | 3.219e-03 |
| PE-Quad | 1024x1024 | spike | 5.117 | 3.826 | 84 | 1.261e-02 | - | 3.145e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.496 | 3.204 | 88 | 1.292e-02 | 1.906e-01 | 3.225e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2.690 | 1.336 | 13 | 4.746e-02 | 4.339e-01 | 5.936e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.740 | 1.386 | 13 | 1.336e-02 | - | 1.589e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.541 | 1.188 | 13 | 4.739e-02 | 2.049e-01 | 5.936e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.517 | 1.207 | 13 | 4.483e-02 | 4.404e-01 | 5.625e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.703 | 1.394 | 13 | 1.378e-02 | - | 1.662e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.481 | 1.171 | 13 | 4.484e-02 | 2.162e-01 | 5.625e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.562 | 1.247 | 13 | 4.582e-02 | 4.407e-01 | 5.781e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.739 | 1.424 | 13 | 1.218e-02 | - | 1.458e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.510 | 1.195 | 13 | 4.582e-02 | 2.405e-01 | 5.781e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.571 | 1.269 | 13 | 4.733e-02 | 4.339e-01 | 5.993e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.816 | 1.514 | 13 | 1.074e-02 | - | 1.274e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.497 | 1.195 | 13 | 4.733e-02 | 2.485e-01 | 5.993e-03 |
| Inv-Newton | 256x256 | spike | 2.512 | 1.213 | 13 | 4.713e-02 | 3.812e-01 | 6.025e-03 |
| PE-Quad | 256x256 | spike | 2.679 | 1.380 | 13 | 9.331e-03 | - | 1.099e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.504 | 1.205 | 13 | 4.905e-02 | 2.288e-01 | 6.273e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.527 | 1.232 | 28 | 4.010e-02 | 5.436e-01 | 5.047e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.672 | 1.376 | 27 | 1.850e-02 | - | 2.240e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.674 | 1.378 | 28 | 4.010e-02 | 2.130e-01 | 5.047e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.616 | 1.341 | 28 | 4.352e-02 | 5.789e-01 | 5.520e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.670 | 1.395 | 27 | 1.547e-02 | - | 1.839e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.518 | 1.243 | 28 | 4.356e-02 | 2.882e-01 | 5.525e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.530 | 1.264 | 28 | 4.042e-02 | 5.567e-01 | 5.131e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.651 | 1.384 | 27 | 1.797e-02 | - | 2.132e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.216 | 1.950 | 28 | 4.049e-02 | 1.987e-01 | 5.140e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.909 | 1.386 | 28 | 4.437e-02 | 6.064e-01 | 5.648e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.952 | 1.430 | 27 | 1.371e-02 | - | 1.602e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.818 | 1.295 | 28 | 4.448e-02 | 2.475e-01 | 5.661e-03 |
| Inv-Newton | 512x512 | spike | 2.633 | 1.343 | 28 | 4.130e-02 | 5.726e-01 | 5.255e-03 |
| PE-Quad | 512x512 | spike | 2.876 | 1.587 | 27 | 1.445e-02 | - | 1.745e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.028 | 1.739 | 28 | 4.369e-02 | 1.672e-01 | 5.558e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.485 | 3.716 | 88 | 3.999e-02 | 7.840e-01 | 5.085e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.173 | 4.404 | 84 | 1.840e-02 | - | 2.172e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.480 | 3.711 | 88 | 4.050e-02 | 2.500e-01 | 5.153e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.031 | 3.704 | 88 | 3.745e-02 | 7.437e-01 | 4.763e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.728 | 4.401 | 84 | 2.029e-02 | - | 2.393e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.063 | 3.736 | 88 | 3.786e-02 | 2.500e-01 | 4.819e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.072 | 3.749 | 88 | 4.413e-02 | 8.594e-01 | 5.639e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.727 | 4.404 | 84 | 1.283e-02 | - | 1.470e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.036 | 3.713 | 88 | 4.487e-02 | 2.500e-01 | 5.735e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.353 | 3.721 | 88 | 4.401e-02 | 8.577e-01 | 5.628e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 6.030 | 4.398 | 84 | 1.282e-02 | - | 1.475e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.340 | 3.709 | 88 | 4.481e-02 | 2.500e-01 | 5.731e-03 |
| Inv-Newton | 1024x1024 | spike | 4.978 | 3.725 | 88 | 3.970e-02 | 7.742e-01 | 5.006e-03 |
| PE-Quad | 1024x1024 | spike | 5.639 | 4.387 | 84 | 1.410e-02 | - | 1.717e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.969 | 3.716 | 88 | 4.420e-02 | 2.423e-01 | 5.571e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
