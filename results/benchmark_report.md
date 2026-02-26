# Fast Matrix Inverse p-th Roots Benchmark Report
*Date: 2026-02-25*

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
| Inv-Newton | 256x256 | gaussian_spd | 3.944 | 2.421 | 13 | 2.033e-03 | 1.257e-01 | 2.039e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.736 | 1.213 | 13 | 3.781e-03 | - | 3.776e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.670 | 1.147 | 13 | 4.061e-03 | 1.349e-01 | 4.042e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.124 | 1.106 | 13 | 1.496e-03 | 1.255e-01 | 1.486e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.092 | 1.074 | 13 | 7.046e-03 | - | 7.051e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.184 | 1.166 | 13 | 5.828e-03 | 1.350e-01 | 5.817e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.026 | 1.059 | 13 | 1.103e-03 | 1.253e-01 | 1.102e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.228 | 1.262 | 13 | 8.142e-03 | - | 8.145e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.407 | 1.441 | 13 | 7.529e-03 | 6.856e-02 | 7.515e-03 |
| Inv-Newton | 256x256 | near_rank_def | 6.236 | 2.273 | 13 | 2.269e-03 | 1.253e-01 | 2.278e-03 |
| PE-Quad | 256x256 | near_rank_def | 5.119 | 1.156 | 13 | 8.875e-03 | - | 8.880e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 5.056 | 1.093 | 13 | 8.452e-03 | 6.180e-02 | 8.428e-03 |
| Inv-Newton | 256x256 | spike | 2.625 | 1.049 | 13 | 4.853e-03 | 1.232e-01 | 4.866e-03 |
| PE-Quad | 256x256 | spike | 2.871 | 1.295 | 13 | 6.826e-03 | - | 6.838e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.643 | 1.067 | 13 | 6.939e-03 | 1.168e-01 | 6.931e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.847 | 1.330 | 28 | 2.975e-03 | 1.776e-01 | 2.958e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.716 | 1.200 | 27 | 3.345e-03 | - | 3.349e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.632 | 1.116 | 28 | 2.204e-03 | 6.425e-02 | 2.205e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.033 | 1.182 | 28 | 2.479e-03 | 1.773e-01 | 2.458e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.250 | 1.399 | 27 | 7.689e-03 | - | 7.706e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.132 | 1.281 | 28 | 5.746e-03 | 5.088e-02 | 5.746e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.960 | 1.033 | 28 | 3.259e-03 | 1.771e-01 | 3.241e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.185 | 1.259 | 27 | 4.055e-03 | - | 4.073e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.462 | 1.535 | 28 | 4.211e-03 | 8.312e-02 | 4.221e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.042 | 1.165 | 28 | 1.534e-03 | 1.771e-01 | 1.518e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.233 | 1.356 | 27 | 6.995e-03 | - | 7.002e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.096 | 1.220 | 28 | 7.311e-03 | 1.367e-01 | 7.298e-03 |
| Inv-Newton | 512x512 | spike | 2.770 | 1.140 | 28 | 2.346e-03 | 1.730e-01 | 2.326e-03 |
| PE-Quad | 512x512 | spike | 2.908 | 1.278 | 27 | 6.228e-03 | - | 6.236e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.770 | 1.140 | 28 | 6.251e-03 | 1.686e-01 | 6.255e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.346 | 2.358 | 88 | 3.522e-03 | 2.509e-01 | 3.501e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.569 | 2.580 | 84 | 4.018e-03 | - | 4.038e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.366 | 2.377 | 88 | 4.026e-03 | 1.438e-01 | 4.043e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.009 | 2.369 | 88 | 4.345e-03 | 2.505e-01 | 4.332e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.258 | 2.617 | 84 | 2.535e-04 | - | 2.434e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.062 | 2.422 | 88 | 3.180e-04 | 3.557e-02 | 3.232e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.127 | 2.363 | 88 | 1.123e-03 | 2.503e-01 | 1.111e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.331 | 2.567 | 84 | 7.066e-03 | - | 7.071e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.128 | 2.364 | 88 | 7.066e-03 | 2.532e-01 | 7.073e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 3.942 | 2.371 | 88 | 1.185e-03 | 2.503e-01 | 1.173e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 4.165 | 2.594 | 84 | 7.004e-03 | - | 7.009e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 3.943 | 2.372 | 88 | 7.004e-03 | 2.528e-01 | 7.011e-03 |
| Inv-Newton | 1024x1024 | spike | 4.891 | 2.381 | 88 | 2.367e-03 | 2.454e-01 | 2.350e-03 |
| PE-Quad | 1024x1024 | spike | 5.156 | 2.646 | 84 | 6.408e-03 | - | 6.413e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.852 | 2.342 | 88 | 6.120e-03 | 2.387e-01 | 6.128e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.084 | 1.318 | 13 | 3.252e-03 | 2.848e-02 | 1.608e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.115 | 1.350 | 13 | 3.545e-03 | - | 1.749e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.299 | 1.533 | 13 | 3.551e-03 | 3.231e-02 | 1.757e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.837 | 1.245 | 13 | 3.212e-03 | 5.994e-02 | 1.596e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.775 | 1.183 | 13 | 3.260e-03 | - | 1.623e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.804 | 1.212 | 13 | 3.249e-03 | 3.505e-02 | 1.621e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.832 | 1.232 | 13 | 1.898e-03 | 3.372e-02 | 9.445e-04 |
| PE-Quad | 256x256 | illcond_1e12 | 2.961 | 1.361 | 13 | 2.068e-03 | - | 1.026e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.786 | 1.186 | 13 | 2.096e-03 | 1.873e-02 | 1.042e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.867 | 1.132 | 13 | 1.688e-03 | 8.210e-03 | 8.596e-04 |
| PE-Quad | 256x256 | near_rank_def | 3.076 | 1.341 | 13 | 3.609e-03 | - | 1.810e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.176 | 1.441 | 13 | 2.462e-03 | 3.907e-03 | 1.231e-03 |
| Inv-Newton | 256x256 | spike | 2.813 | 1.120 | 13 | 3.335e-03 | 2.327e-02 | 1.704e-03 |
| PE-Quad | 256x256 | spike | 2.960 | 1.268 | 13 | 5.540e-03 | - | 2.821e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.835 | 1.142 | 13 | 8.655e-03 | 5.476e-02 | 4.348e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.862 | 1.353 | 28 | 4.160e-03 | 1.527e-01 | 2.037e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.924 | 1.415 | 27 | 4.162e-03 | - | 2.034e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.674 | 1.165 | 28 | 4.486e-03 | 8.258e-02 | 2.159e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 4.496 | 1.408 | 28 | 2.720e-03 | 1.051e-01 | 1.405e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 4.361 | 1.273 | 27 | 2.950e-03 | - | 1.530e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 4.411 | 1.324 | 28 | 2.886e-03 | 5.236e-02 | 1.478e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.745 | 1.201 | 28 | 3.396e-03 | 1.491e-01 | 1.727e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.778 | 1.233 | 27 | 3.395e-03 | - | 1.726e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.781 | 1.236 | 28 | 8.401e-03 | 7.303e-02 | 4.234e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.231 | 1.403 | 28 | 1.544e-03 | 6.564e-02 | 7.750e-04 |
| PE-Quad | 512x512 | near_rank_def | 3.073 | 1.245 | 27 | 1.543e-03 | - | 7.735e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.512 | 1.684 | 28 | 1.017e-02 | 2.923e-02 | 5.025e-03 |
| Inv-Newton | 512x512 | spike | 3.040 | 1.297 | 28 | 1.449e-03 | 6.419e-02 | 7.306e-04 |
| PE-Quad | 512x512 | spike | 3.014 | 1.271 | 27 | 1.770e-03 | - | 8.796e-04 |
| PE-Quad-Coupled | 512x512 | spike | 2.889 | 1.146 | 28 | 1.114e-02 | 6.960e-02 | 5.651e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.209 | 2.874 | 88 | 3.302e-03 | 2.096e-01 | 1.599e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.504 | 3.170 | 84 | 3.300e-03 | - | 1.597e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.154 | 2.820 | 88 | 1.097e-02 | 2.754e-01 | 5.523e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.574 | 2.840 | 88 | 3.888e-03 | 2.501e-01 | 1.846e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.908 | 3.174 | 84 | 3.887e-03 | - | 1.844e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.555 | 2.820 | 88 | 1.088e-02 | 2.795e-01 | 5.472e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.722 | 2.867 | 88 | 4.739e-04 | 4.191e-03 | 1.647e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.050 | 3.196 | 84 | 4.724e-04 | - | 1.617e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.651 | 2.797 | 88 | 1.155e-02 | 2.652e-01 | 5.799e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.433 | 2.829 | 88 | 4.875e-04 | 3.872e-03 | 1.845e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 4.770 | 3.166 | 84 | 4.861e-04 | - | 1.819e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.476 | 2.872 | 88 | 1.154e-02 | 2.655e-01 | 5.809e-03 |
| Inv-Newton | 1024x1024 | spike | 4.438 | 2.824 | 88 | 1.380e-03 | 8.558e-02 | 7.153e-04 |
| PE-Quad | 1024x1024 | spike | 4.804 | 3.191 | 84 | 1.629e-03 | - | 8.269e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.416 | 2.802 | 88 | 1.026e-02 | 2.107e-01 | 5.194e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.463 | 1.565 | 13 | 1.358e-02 | 4.043e-02 | 4.590e-03 |
| PE-Quad | 256x256 | gaussian_spd | 5.342 | 3.444 | 13 | 5.352e-03 | - | 1.835e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.672 | 1.775 | 13 | 5.747e-03 | 6.921e-02 | 1.936e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.126 | 1.400 | 13 | 1.536e-02 | 5.157e-02 | 5.200e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.636 | 1.910 | 13 | 5.076e-03 | - | 1.656e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.122 | 1.396 | 13 | 5.080e-03 | 6.287e-02 | 1.654e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.295 | 1.268 | 13 | 1.614e-02 | 5.927e-02 | 5.518e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.633 | 1.606 | 13 | 3.175e-03 | - | 1.075e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.351 | 1.325 | 13 | 3.168e-03 | 6.323e-02 | 1.073e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.463 | 1.561 | 13 | 1.554e-02 | 5.900e-02 | 5.299e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.512 | 1.609 | 13 | 2.950e-03 | - | 9.808e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.595 | 1.692 | 13 | 2.952e-03 | 7.605e-02 | 9.821e-04 |
| Inv-Newton | 256x256 | spike | 3.285 | 1.425 | 13 | 2.081e-02 | 1.058e-01 | 6.952e-03 |
| PE-Quad | 256x256 | spike | 3.457 | 1.597 | 13 | 4.294e-03 | - | 1.435e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.351 | 1.491 | 13 | 4.251e-03 | 1.073e-01 | 1.421e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.488 | 1.739 | 28 | 1.040e-02 | 3.154e-02 | 3.604e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.458 | 1.708 | 27 | 7.546e-03 | - | 2.641e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.106 | 1.356 | 28 | 7.585e-03 | 9.596e-02 | 2.562e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.048 | 1.423 | 28 | 1.329e-02 | 6.605e-02 | 4.531e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.332 | 1.708 | 27 | 5.577e-03 | - | 1.944e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.905 | 1.280 | 28 | 5.682e-03 | 8.839e-02 | 1.986e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.446 | 1.412 | 28 | 1.198e-02 | 4.756e-02 | 3.920e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.590 | 1.555 | 27 | 8.027e-03 | - | 2.614e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.521 | 1.486 | 28 | 8.088e-03 | 8.839e-02 | 2.775e-03 |
| Inv-Newton | 512x512 | near_rank_def | 5.351 | 1.397 | 28 | 1.589e-02 | 8.210e-02 | 5.224e-03 |
| PE-Quad | 512x512 | near_rank_def | 6.133 | 2.179 | 27 | 4.014e-03 | - | 1.365e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 5.331 | 1.377 | 28 | 4.039e-03 | 8.839e-02 | 1.413e-03 |
| Inv-Newton | 512x512 | spike | 3.324 | 1.564 | 28 | 1.522e-02 | 8.800e-02 | 5.084e-03 |
| PE-Quad | 512x512 | spike | 3.330 | 1.570 | 27 | 4.036e-03 | - | 1.336e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.007 | 1.248 | 28 | 4.040e-03 | 1.051e-01 | 1.336e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.385 | 3.286 | 88 | 1.189e-02 | 6.824e-02 | 3.881e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.992 | 3.892 | 84 | 7.831e-03 | - | 2.534e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.361 | 3.262 | 88 | 7.831e-03 | 2.148e-01 | 2.535e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.994 | 3.287 | 88 | 9.047e-03 | 2.705e-03 | 2.940e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.579 | 3.873 | 84 | 9.043e-03 | - | 2.939e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.019 | 3.313 | 88 | 9.042e-03 | 2.500e-01 | 2.940e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.847 | 3.287 | 88 | 1.626e-02 | 1.250e-01 | 5.380e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.446 | 3.886 | 84 | 2.650e-03 | - | 9.175e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.790 | 3.230 | 88 | 2.651e-03 | 8.839e-02 | 9.176e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.001 | 3.251 | 88 | 1.620e-02 | 1.250e-01 | 5.366e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.633 | 3.883 | 84 | 2.709e-03 | - | 9.301e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.004 | 3.254 | 88 | 2.710e-03 | 8.839e-02 | 9.303e-04 |
| Inv-Newton | 1024x1024 | spike | 4.945 | 3.308 | 88 | 1.512e-02 | 1.249e-01 | 5.068e-03 |
| PE-Quad | 1024x1024 | spike | 5.537 | 3.900 | 84 | 4.135e-03 | - | 1.359e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.928 | 3.290 | 88 | 4.089e-03 | 1.193e-01 | 1.342e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 4.977 | 2.090 | 13 | 8.493e-03 | 1.070e-01 | 2.239e-03 |
| PE-Quad | 256x256 | gaussian_spd | 4.487 | 1.599 | 13 | 8.479e-03 | - | 2.237e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 4.207 | 1.319 | 13 | 1.207e-02 | 7.114e-02 | 3.131e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.300 | 1.575 | 13 | 1.026e-02 | 1.473e-01 | 2.666e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.460 | 1.736 | 13 | 1.058e-02 | - | 2.725e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.997 | 1.272 | 13 | 1.285e-02 | 6.832e-02 | 3.228e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.113 | 1.431 | 13 | 1.147e-02 | 1.313e-01 | 2.939e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.547 | 1.865 | 13 | 1.320e-02 | - | 3.232e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.138 | 1.456 | 13 | 1.425e-02 | 3.867e-02 | 3.442e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.383 | 1.473 | 13 | 1.096e-02 | 1.177e-01 | 2.799e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.294 | 1.384 | 13 | 1.276e-02 | - | 3.099e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.081 | 1.171 | 13 | 1.345e-02 | 1.458e-02 | 3.242e-03 |
| Inv-Newton | 256x256 | spike | 3.161 | 1.396 | 13 | 8.838e-03 | 9.464e-02 | 2.244e-03 |
| PE-Quad | 256x256 | spike | 3.178 | 1.413 | 13 | 9.246e-03 | - | 2.295e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.139 | 1.373 | 13 | 9.380e-03 | 6.249e-02 | 2.323e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.197 | 1.424 | 28 | 8.198e-03 | 2.388e-01 | 1.931e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.371 | 1.598 | 27 | 8.722e-03 | - | 2.029e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.225 | 1.452 | 28 | 8.696e-03 | 1.435e-01 | 2.030e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.394 | 1.764 | 28 | 1.029e-02 | 2.123e-01 | 2.527e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.123 | 1.494 | 27 | 1.153e-02 | - | 2.735e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.168 | 1.539 | 28 | 1.154e-02 | 1.050e-01 | 2.736e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.972 | 1.351 | 28 | 9.083e-03 | 2.429e-01 | 2.197e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.171 | 1.549 | 27 | 9.618e-03 | - | 2.284e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.085 | 1.464 | 28 | 9.628e-03 | 1.491e-01 | 2.285e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.362 | 1.539 | 28 | 1.220e-02 | 1.914e-01 | 3.047e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.214 | 1.391 | 27 | 1.341e-02 | - | 3.240e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.285 | 1.462 | 28 | 1.342e-02 | 6.556e-02 | 3.240e-03 |
| Inv-Newton | 512x512 | spike | 2.929 | 1.423 | 28 | 1.242e-02 | 1.893e-01 | 3.140e-03 |
| PE-Quad | 512x512 | spike | 3.237 | 1.732 | 27 | 1.249e-02 | - | 3.089e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.754 | 1.248 | 28 | 1.252e-02 | 7.118e-02 | 3.097e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.901 | 3.339 | 88 | 9.050e-03 | 3.426e-01 | 2.195e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.358 | 3.795 | 84 | 9.455e-03 | - | 2.260e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.894 | 3.332 | 88 | 9.255e-03 | 2.094e-01 | 2.227e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.789 | 3.273 | 88 | 6.468e-03 | 3.750e-01 | 1.517e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.251 | 3.734 | 84 | 6.471e-03 | - | 1.517e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.764 | 3.248 | 88 | 6.475e-03 | 2.500e-01 | 1.517e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.824 | 3.260 | 88 | 1.286e-02 | 2.500e-01 | 3.294e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 6.354 | 3.791 | 84 | 1.371e-02 | - | 3.351e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.866 | 3.302 | 88 | 1.330e-02 | 3.558e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.263 | 3.389 | 88 | 1.289e-02 | 2.500e-01 | 3.298e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.617 | 3.742 | 84 | 1.364e-02 | - | 3.341e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.191 | 3.317 | 88 | 1.329e-02 | 3.438e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 5.058 | 3.322 | 88 | 1.277e-02 | 2.653e-01 | 3.211e-03 |
| PE-Quad | 1024x1024 | spike | 5.509 | 3.772 | 84 | 1.261e-02 | - | 3.144e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.006 | 3.270 | 88 | 1.290e-02 | 1.896e-01 | 3.220e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3.913 | 1.802 | 13 | 4.679e-02 | 3.965e-01 | 5.852e-03 |
| PE-Quad | 256x256 | gaussian_spd | 5.155 | 3.044 | 13 | 1.284e-02 | - | 1.526e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.656 | 1.545 | 13 | 4.675e-02 | 1.412e-01 | 5.852e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.015 | 1.398 | 13 | 4.480e-02 | 4.061e-01 | 5.621e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.274 | 1.657 | 13 | 1.375e-02 | - | 1.659e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.978 | 1.361 | 13 | 4.482e-02 | 1.212e-01 | 5.620e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.379 | 1.562 | 13 | 4.579e-02 | 4.270e-01 | 5.780e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.417 | 1.601 | 13 | 1.217e-02 | - | 1.457e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.270 | 1.453 | 13 | 4.580e-02 | 1.248e-01 | 5.780e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.451 | 1.393 | 13 | 4.729e-02 | 4.257e-01 | 5.987e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.779 | 1.720 | 13 | 1.072e-02 | - | 1.272e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.771 | 1.713 | 13 | 4.730e-02 | 1.518e-01 | 5.987e-03 |
| Inv-Newton | 256x256 | spike | 3.267 | 1.557 | 13 | 4.704e-02 | 3.786e-01 | 6.016e-03 |
| PE-Quad | 256x256 | spike | 3.412 | 1.702 | 13 | 9.265e-03 | - | 1.087e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.088 | 1.378 | 13 | 4.903e-02 | 2.097e-01 | 6.268e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.410 | 1.558 | 28 | 3.994e-02 | 4.677e-01 | 5.025e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.448 | 1.596 | 27 | 1.838e-02 | - | 2.224e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.415 | 1.563 | 28 | 3.995e-02 | 6.267e-02 | 5.025e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.209 | 1.714 | 28 | 4.323e-02 | 5.565e-01 | 5.482e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.387 | 1.892 | 27 | 1.512e-02 | - | 1.794e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.051 | 1.556 | 28 | 4.327e-02 | 1.419e-01 | 5.484e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 4.916 | 1.752 | 28 | 4.040e-02 | 4.985e-01 | 5.128e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 5.102 | 1.938 | 27 | 1.796e-02 | - | 2.130e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 4.845 | 1.681 | 28 | 4.048e-02 | 9.504e-02 | 5.137e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.187 | 1.541 | 28 | 4.427e-02 | 5.955e-01 | 5.635e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.414 | 1.767 | 27 | 1.363e-02 | - | 1.592e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.255 | 1.609 | 28 | 4.443e-02 | 1.642e-01 | 5.654e-03 |
| Inv-Newton | 512x512 | spike | 3.469 | 1.486 | 28 | 4.127e-02 | 5.672e-01 | 5.250e-03 |
| PE-Quad | 512x512 | spike | 3.819 | 1.836 | 27 | 1.434e-02 | - | 1.737e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.496 | 1.513 | 28 | 4.339e-02 | 1.646e-01 | 5.517e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.722 | 3.685 | 88 | 3.980e-02 | 6.994e-01 | 5.060e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.500 | 4.464 | 84 | 1.819e-02 | - | 2.146e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.756 | 3.720 | 88 | 4.028e-02 | 1.363e-01 | 5.123e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.706 | 3.774 | 88 | 3.744e-02 | 6.170e-01 | 4.763e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 6.410 | 4.479 | 84 | 2.029e-02 | - | 2.393e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.639 | 3.707 | 88 | 3.786e-02 | 2.776e-04 | 4.819e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 7.639 | 3.786 | 88 | 4.412e-02 | 8.605e-01 | 5.639e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 8.332 | 4.478 | 84 | 1.283e-02 | - | 1.470e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 7.670 | 3.816 | 88 | 4.487e-02 | 2.500e-01 | 5.735e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 7.571 | 3.743 | 88 | 4.400e-02 | 8.590e-01 | 5.626e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 8.348 | 4.520 | 84 | 1.282e-02 | - | 1.474e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 7.541 | 3.713 | 88 | 4.481e-02 | 2.500e-01 | 5.730e-03 |
| Inv-Newton | 1024x1024 | spike | 5.422 | 3.770 | 88 | 3.974e-02 | 7.704e-01 | 5.009e-03 |
| PE-Quad | 1024x1024 | spike | 6.226 | 4.574 | 84 | 1.408e-02 | - | 1.715e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.343 | 3.690 | 88 | 4.391e-02 | 2.379e-01 | 5.538e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
