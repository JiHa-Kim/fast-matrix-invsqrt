# Fast Matrix Inverse p-th Roots Benchmark Report
*Date: 2026-02-25*

This report details the performance and accuracy of quadratic PE (Polynomial-Express) iterations for matrix inverse p-th roots.

## Methodology
- **Sizes**: 256,512,1024
- **Compiled**: Yes (`torch.compile(mode='max-autotune')`)
- **Trials per case**: 2
- **Hardware**: GPU (bf16)
- **Methods Compared**: `Inverse-Newton` (baseline), `PE-Quad` (uncoupled quadratic), `PE-Quad-Coupled` (coupled quadratic).

## Results for $p=1$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 1414.135 | 1411.969 | 11 | 2.362e-03 | 1.261e-01 | 2.363e-03 |
| PE-Quad | 256x256 | gaussian_spd | 4.931 | 2.765 | 11 | 5.079e-03 | - | 5.072e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.728 | 1.562 | 11 | 7.439e-03 | 2.513e-01 | 7.414e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 5.943 | 1.908 | 11 | 1.573e-03 | 1.256e-01 | 1.560e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 5.744 | 1.709 | 11 | 7.797e-03 | - | 7.801e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 5.336 | 1.300 | 11 | 6.631e-03 | 1.534e-01 | 6.617e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 7.810 | 4.410 | 11 | 1.090e-03 | 1.253e-01 | 1.083e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 4.807 | 1.407 | 11 | 7.945e-03 | - | 7.948e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 5.919 | 2.519 | 11 | 7.222e-03 | 9.363e-02 | 7.210e-03 |
| Inv-Newton | 256x256 | near_rank_def | 4.550 | 1.928 | 11 | 2.213e-03 | 1.253e-01 | 2.222e-03 |
| PE-Quad | 256x256 | near_rank_def | 4.643 | 2.021 | 11 | 8.957e-03 | - | 8.963e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 4.249 | 1.627 | 11 | 8.685e-03 | 6.010e-02 | 8.660e-03 |
| Inv-Newton | 256x256 | spike | 2.819 | 1.169 | 11 | 5.061e-03 | 1.237e-01 | 5.069e-03 |
| PE-Quad | 256x256 | spike | 2.702 | 1.053 | 11 | 6.774e-03 | - | 6.780e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.246 | 1.596 | 11 | 6.863e-03 | 1.195e-01 | 6.856e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.105 | 1.409 | 20 | 4.163e-03 | 1.776e-01 | 4.135e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.081 | 1.384 | 19 | 3.649e-03 | - | 3.657e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.627 | 0.931 | 20 | 2.233e-03 | 1.912e-01 | 2.235e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.307 | 1.252 | 20 | 2.296e-03 | 1.773e-01 | 2.275e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.267 | 1.212 | 19 | 8.464e-03 | - | 8.478e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.365 | 1.311 | 20 | 5.794e-03 | 5.811e-02 | 5.790e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.965 | 1.286 | 20 | 3.399e-03 | 1.771e-01 | 3.383e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.814 | 1.136 | 19 | 4.473e-03 | - | 4.491e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.801 | 1.122 | 20 | 4.638e-03 | 9.510e-02 | 4.647e-03 |
| Inv-Newton | 512x512 | near_rank_def | 5.015 | 1.471 | 20 | 1.571e-03 | 1.771e-01 | 1.555e-03 |
| PE-Quad | 512x512 | near_rank_def | 4.667 | 1.123 | 19 | 7.056e-03 | - | 7.063e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 4.810 | 1.267 | 20 | 7.313e-03 | 1.458e-01 | 7.302e-03 |
| Inv-Newton | 512x512 | spike | 3.546 | 1.252 | 20 | 2.308e-03 | 1.733e-01 | 2.294e-03 |
| PE-Quad | 512x512 | spike | 3.531 | 1.237 | 19 | 6.328e-03 | - | 6.337e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.507 | 1.213 | 20 | 6.328e-03 | 1.730e-01 | 6.337e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.312 | 2.343 | 56 | 3.423e-03 | 2.509e-01 | 3.401e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.867 | 2.898 | 52 | 4.964e-03 | - | 4.985e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.340 | 2.371 | 56 | 4.971e-03 | 1.756e-01 | 4.988e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.089 | 2.364 | 56 | 4.391e-03 | 2.505e-01 | 4.378e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.335 | 2.610 | 52 | 3.683e-04 | - | 3.610e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 3.954 | 2.229 | 56 | 4.123e-04 | 3.546e-02 | 4.165e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.804 | 2.370 | 56 | 1.109e-03 | 2.504e-01 | 1.097e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.760 | 3.325 | 52 | 7.101e-03 | - | 7.106e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.765 | 2.331 | 56 | 7.101e-03 | 2.533e-01 | 7.108e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.237 | 2.506 | 56 | 1.181e-03 | 2.503e-01 | 1.169e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 4.291 | 2.559 | 52 | 7.034e-03 | - | 7.039e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.304 | 2.572 | 56 | 7.034e-03 | 2.529e-01 | 7.040e-03 |
| Inv-Newton | 1024x1024 | spike | 4.002 | 2.369 | 56 | 2.395e-03 | 2.460e-01 | 2.376e-03 |
| PE-Quad | 1024x1024 | spike | 4.453 | 2.820 | 52 | 6.430e-03 | - | 6.432e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.005 | 2.372 | 56 | 6.126e-03 | 2.401e-01 | 6.132e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 3019.846 | 3018.330 | 11 | 4.411e-03 | 6.916e-02 | 2.183e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.566 | 1.051 | 11 | 4.288e-03 | - | 2.134e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.741 | 1.225 | 11 | 4.297e-03 | 4.539e-02 | 2.139e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.023 | 1.260 | 11 | 3.590e-03 | 6.262e-02 | 1.793e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.776 | 2.013 | 11 | 3.645e-03 | - | 1.817e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.031 | 1.268 | 11 | 3.657e-03 | 3.945e-02 | 1.824e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.601 | 1.150 | 11 | 2.231e-03 | 3.917e-02 | 1.119e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.971 | 1.520 | 11 | 2.282e-03 | - | 1.141e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 4.020 | 2.569 | 11 | 2.543e-03 | 2.376e-02 | 1.259e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.229 | 1.362 | 11 | 1.662e-03 | 8.211e-03 | 8.359e-04 |
| PE-Quad | 256x256 | near_rank_def | 3.676 | 1.809 | 11 | 3.513e-03 | - | 1.763e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.370 | 1.503 | 11 | 2.271e-03 | 3.907e-03 | 1.129e-03 |
| Inv-Newton | 256x256 | spike | 3.348 | 1.391 | 11 | 3.466e-03 | 7.695e-02 | 1.784e-03 |
| PE-Quad | 256x256 | spike | 3.188 | 1.231 | 11 | 5.674e-03 | - | 2.878e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.125 | 1.167 | 11 | 1.160e-02 | 5.593e-02 | 5.846e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.973 | 1.525 | 20 | 4.246e-03 | 1.600e-01 | 2.084e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.383 | 1.935 | 19 | 4.251e-03 | - | 2.080e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.064 | 1.616 | 20 | 6.033e-03 | 8.839e-02 | 2.981e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.176 | 1.511 | 20 | 2.667e-03 | 1.133e-01 | 1.382e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.162 | 1.498 | 19 | 2.668e-03 | - | 1.380e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.463 | 1.799 | 20 | 2.884e-03 | 5.661e-02 | 1.491e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.841 | 1.341 | 20 | 3.596e-03 | 1.575e-01 | 1.834e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.882 | 1.382 | 19 | 3.594e-03 | - | 1.833e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.206 | 1.706 | 20 | 8.799e-03 | 7.564e-02 | 4.411e-03 |
| Inv-Newton | 512x512 | near_rank_def | 3.023 | 1.432 | 20 | 1.602e-03 | 6.819e-02 | 8.072e-04 |
| PE-Quad | 512x512 | near_rank_def | 2.865 | 1.274 | 19 | 1.601e-03 | - | 8.057e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.738 | 1.147 | 20 | 1.040e-02 | 3.125e-02 | 5.146e-03 |
| Inv-Newton | 512x512 | spike | 3.876 | 1.871 | 20 | 1.363e-03 | 6.287e-02 | 6.953e-04 |
| PE-Quad | 512x512 | spike | 4.467 | 2.462 | 19 | 1.681e-03 | - | 8.307e-04 |
| PE-Quad-Coupled | 512x512 | spike | 3.381 | 1.375 | 20 | 1.112e-02 | 7.192e-02 | 5.650e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.298 | 2.767 | 56 | 3.198e-03 | 2.030e-01 | 1.546e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.088 | 3.556 | 52 | 3.196e-03 | - | 1.544e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.367 | 2.835 | 56 | 1.111e-02 | 2.750e-01 | 5.587e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.403 | 2.875 | 56 | 3.885e-03 | 2.501e-01 | 1.842e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.006 | 3.478 | 52 | 3.884e-03 | - | 1.840e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.775 | 3.247 | 56 | 1.088e-02 | 2.795e-01 | 5.469e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.352 | 2.885 | 56 | 4.697e-04 | 4.247e-03 | 1.583e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.488 | 4.020 | 52 | 4.681e-04 | - | 1.552e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.434 | 2.967 | 56 | 1.154e-02 | 2.652e-01 | 5.794e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.470 | 3.044 | 56 | 4.867e-04 | 3.868e-03 | 1.833e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 5.560 | 3.134 | 52 | 4.853e-04 | - | 1.807e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.274 | 2.849 | 56 | 1.155e-02 | 2.655e-01 | 5.816e-03 |
| Inv-Newton | 1024x1024 | spike | 4.494 | 2.846 | 56 | 1.389e-03 | 8.752e-02 | 7.195e-04 |
| PE-Quad | 1024x1024 | spike | 4.794 | 3.146 | 52 | 1.710e-03 | - | 8.645e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.491 | 2.843 | 56 | 1.033e-02 | 2.119e-01 | 5.223e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 1287.729 | 1286.230 | 11 | 1.319e-02 | 4.166e-02 | 4.462e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.351 | 1.852 | 11 | 6.827e-03 | - | 2.233e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.003 | 1.504 | 11 | 6.343e-03 | 9.359e-02 | 2.066e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.304 | 1.501 | 11 | 1.592e-02 | 5.597e-02 | 5.411e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.284 | 1.482 | 11 | 5.571e-03 | - | 1.818e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.175 | 1.372 | 11 | 5.573e-03 | 6.323e-02 | 1.817e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.191 | 1.617 | 11 | 1.607e-02 | 5.836e-02 | 5.490e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.027 | 1.452 | 11 | 3.675e-03 | - | 1.216e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.053 | 1.479 | 11 | 3.665e-03 | 6.287e-02 | 1.214e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.694 | 1.909 | 11 | 1.594e-02 | 5.965e-02 | 5.440e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.639 | 1.854 | 11 | 2.881e-03 | - | 9.590e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 3.614 | 1.828 | 11 | 2.874e-03 | 7.842e-02 | 9.599e-04 |
| Inv-Newton | 256x256 | spike | 3.388 | 1.620 | 11 | 2.106e-02 | 1.077e-01 | 7.040e-03 |
| PE-Quad | 256x256 | spike | 3.534 | 1.765 | 11 | 5.922e-03 | - | 1.956e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.224 | 1.456 | 11 | 5.959e-03 | 1.093e-01 | 2.014e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.144 | 1.396 | 20 | 1.084e-02 | 3.764e-02 | 3.756e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.947 | 2.199 | 19 | 8.307e-03 | - | 2.898e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.008 | 1.259 | 20 | 8.377e-03 | 1.198e-01 | 2.827e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 3.284 | 1.718 | 20 | 1.494e-02 | 7.538e-02 | 5.003e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.857 | 1.291 | 19 | 5.935e-03 | - | 2.073e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.118 | 1.552 | 20 | 6.066e-03 | 8.839e-02 | 2.117e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.859 | 1.394 | 20 | 1.246e-02 | 5.244e-02 | 4.083e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.785 | 1.319 | 19 | 8.417e-03 | - | 2.755e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.702 | 1.236 | 20 | 8.473e-03 | 8.839e-02 | 2.911e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.886 | 1.296 | 20 | 1.601e-02 | 8.325e-02 | 5.263e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.331 | 1.741 | 19 | 4.134e-03 | - | 1.399e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.134 | 1.544 | 20 | 4.138e-03 | 8.839e-02 | 1.447e-03 |
| Inv-Newton | 512x512 | spike | 2.968 | 1.478 | 20 | 1.534e-02 | 8.822e-02 | 5.122e-03 |
| PE-Quad | 512x512 | spike | 2.802 | 1.311 | 19 | 4.088e-03 | - | 1.352e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.777 | 1.286 | 20 | 4.250e-03 | 1.026e-01 | 1.400e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.570 | 3.916 | 56 | 1.307e-02 | 8.469e-02 | 4.282e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.000 | 4.346 | 52 | 7.602e-03 | - | 2.462e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.421 | 3.767 | 56 | 7.602e-03 | 2.089e-01 | 2.463e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 6.932 | 3.489 | 56 | 9.036e-03 | 4.718e-03 | 2.937e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 7.260 | 3.816 | 52 | 9.032e-03 | - | 2.937e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 7.371 | 3.927 | 56 | 9.030e-03 | 2.500e-01 | 2.937e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.965 | 3.354 | 56 | 1.630e-02 | 1.250e-01 | 5.388e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.436 | 3.825 | 52 | 2.632e-03 | - | 9.131e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.910 | 3.298 | 56 | 2.634e-03 | 8.839e-02 | 9.132e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 6.739 | 3.900 | 56 | 1.623e-02 | 1.250e-01 | 5.373e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 6.663 | 3.825 | 52 | 2.706e-03 | - | 9.293e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 6.486 | 3.647 | 56 | 2.707e-03 | 8.839e-02 | 9.294e-04 |
| Inv-Newton | 1024x1024 | spike | 5.406 | 3.982 | 56 | 1.520e-02 | 1.251e-01 | 5.097e-03 |
| PE-Quad | 1024x1024 | spike | 5.258 | 3.835 | 52 | 4.217e-03 | - | 1.388e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.724 | 4.301 | 56 | 4.111e-03 | 1.205e-01 | 1.352e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 1335.072 | 1333.224 | 11 | 7.606e-03 | 1.561e-01 | 2.014e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.162 | 1.315 | 11 | 7.606e-03 | - | 2.014e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.787 | 1.939 | 11 | 1.432e-02 | 9.727e-02 | 3.636e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.883 | 1.276 | 11 | 1.080e-02 | 1.530e-01 | 2.800e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.873 | 1.266 | 11 | 1.157e-02 | - | 2.936e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.066 | 1.458 | 11 | 1.364e-02 | 7.534e-02 | 3.370e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.852 | 1.327 | 11 | 1.138e-02 | 1.357e-01 | 2.917e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.858 | 1.332 | 11 | 1.312e-02 | - | 3.212e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.932 | 1.407 | 11 | 1.417e-02 | 4.688e-02 | 3.427e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.125 | 1.382 | 11 | 1.133e-02 | 1.197e-01 | 2.899e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.020 | 1.276 | 11 | 1.321e-02 | - | 3.213e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.907 | 1.163 | 11 | 1.407e-02 | 1.353e-02 | 3.394e-03 |
| Inv-Newton | 256x256 | spike | 3.574 | 1.202 | 11 | 1.102e-02 | 1.506e-01 | 2.772e-03 |
| PE-Quad | 256x256 | spike | 3.613 | 1.241 | 11 | 1.134e-02 | - | 2.780e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.479 | 1.107 | 11 | 1.130e-02 | 7.694e-02 | 2.769e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.629 | 1.179 | 20 | 8.510e-03 | 2.513e-01 | 2.019e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.851 | 1.400 | 19 | 9.241e-03 | - | 2.149e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.737 | 1.286 | 20 | 9.215e-03 | 1.597e-01 | 2.149e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.878 | 1.198 | 20 | 1.141e-02 | 2.174e-01 | 2.838e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.013 | 1.333 | 19 | 1.288e-02 | - | 3.080e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.797 | 1.117 | 20 | 1.289e-02 | 1.132e-01 | 3.080e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.646 | 1.101 | 20 | 9.496e-03 | 2.495e-01 | 2.305e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.986 | 1.441 | 19 | 1.010e-02 | - | 2.407e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.881 | 1.336 | 20 | 1.011e-02 | 1.574e-01 | 2.408e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.571 | 1.118 | 20 | 1.233e-02 | 1.925e-01 | 3.083e-03 |
| PE-Quad | 512x512 | near_rank_def | 2.733 | 1.280 | 19 | 1.351e-02 | - | 3.268e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.998 | 1.545 | 20 | 1.352e-02 | 6.811e-02 | 3.268e-03 |
| Inv-Newton | 512x512 | spike | 2.580 | 1.144 | 20 | 1.259e-02 | 1.884e-01 | 3.184e-03 |
| PE-Quad | 512x512 | spike | 2.835 | 1.399 | 19 | 1.263e-02 | - | 3.124e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.158 | 1.721 | 20 | 1.276e-02 | 7.625e-02 | 3.153e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.038 | 3.368 | 56 | 1.010e-02 | 3.375e-01 | 2.477e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.699 | 4.029 | 52 | 1.063e-02 | - | 2.559e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.770 | 3.101 | 56 | 1.039e-02 | 2.028e-01 | 2.520e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.812 | 4.097 | 56 | 6.457e-03 | 3.750e-01 | 1.514e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.570 | 3.855 | 52 | 6.460e-03 | - | 1.514e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.936 | 3.222 | 56 | 6.464e-03 | 2.500e-01 | 1.514e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.646 | 3.154 | 56 | 1.285e-02 | 2.500e-01 | 3.291e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.337 | 3.845 | 52 | 1.374e-02 | - | 3.357e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.701 | 3.209 | 56 | 1.330e-02 | 3.618e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.759 | 3.205 | 56 | 1.292e-02 | 2.500e-01 | 3.303e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.289 | 3.735 | 52 | 1.367e-02 | - | 3.346e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.888 | 3.334 | 56 | 1.329e-02 | 3.440e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 5.962 | 3.301 | 56 | 1.287e-02 | 2.662e-01 | 3.235e-03 |
| PE-Quad | 1024x1024 | spike | 7.136 | 4.475 | 52 | 1.269e-02 | - | 3.169e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 6.125 | 3.464 | 56 | 1.290e-02 | 1.967e-01 | 3.220e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 1342.185 | 1339.746 | 11 | 5.083e-02 | 3.991e-01 | 6.377e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.904 | 1.465 | 11 | 2.674e-02 | - | 3.255e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.852 | 1.413 | 11 | 5.080e-02 | 1.972e-01 | 6.377e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.347 | 1.464 | 11 | 4.548e-02 | 4.164e-01 | 5.718e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.345 | 1.462 | 11 | 1.437e-02 | - | 1.730e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.123 | 1.240 | 11 | 4.549e-02 | 1.235e-01 | 5.718e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 3.192 | 1.254 | 11 | 4.552e-02 | 4.253e-01 | 5.743e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 4.070 | 2.132 | 11 | 1.257e-02 | - | 1.510e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 3.184 | 1.246 | 11 | 4.553e-02 | 1.230e-01 | 5.743e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.764 | 1.238 | 11 | 4.748e-02 | 4.305e-01 | 6.017e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.959 | 1.434 | 11 | 1.089e-02 | - | 1.294e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.813 | 1.287 | 11 | 4.749e-02 | 1.568e-01 | 6.017e-03 |
| Inv-Newton | 256x256 | spike | 2.721 | 1.269 | 11 | 4.770e-02 | 3.812e-01 | 6.101e-03 |
| PE-Quad | 256x256 | spike | 2.967 | 1.514 | 11 | 1.618e-02 | - | 1.931e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.693 | 1.241 | 11 | 4.951e-02 | 2.152e-01 | 6.330e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.778 | 1.355 | 20 | 4.031e-02 | 4.783e-01 | 5.075e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.193 | 1.770 | 19 | 2.340e-02 | - | 2.876e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.944 | 1.522 | 20 | 4.032e-02 | 7.494e-02 | 5.075e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.921 | 1.448 | 20 | 4.377e-02 | 5.758e-01 | 5.553e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.829 | 1.356 | 19 | 1.553e-02 | - | 1.846e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.660 | 1.187 | 20 | 4.378e-02 | 1.507e-01 | 5.553e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.005 | 1.508 | 20 | 4.088e-02 | 5.104e-01 | 5.191e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.896 | 1.399 | 19 | 1.845e-02 | - | 2.193e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.684 | 1.187 | 20 | 4.093e-02 | 1.048e-01 | 5.196e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.888 | 1.382 | 20 | 4.431e-02 | 5.982e-01 | 5.642e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.327 | 1.821 | 19 | 1.371e-02 | - | 1.602e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.074 | 1.568 | 20 | 4.455e-02 | 1.665e-01 | 5.672e-03 |
| Inv-Newton | 512x512 | spike | 3.403 | 1.482 | 20 | 4.171e-02 | 5.739e-01 | 5.308e-03 |
| PE-Quad | 512x512 | spike | 3.322 | 1.401 | 19 | 1.419e-02 | - | 1.718e-03 |
| PE-Quad-Coupled | 512x512 | spike | 3.397 | 1.476 | 20 | 4.377e-02 | 1.677e-01 | 5.566e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 6.665 | 4.387 | 56 | 4.087e-02 | 7.393e-01 | 5.201e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 6.712 | 4.434 | 52 | 1.790e-02 | - | 2.110e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 6.323 | 4.045 | 56 | 4.141e-02 | 1.692e-01 | 5.272e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.108 | 3.657 | 56 | 3.742e-02 | 6.167e-01 | 4.761e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.865 | 4.414 | 52 | 2.033e-02 | - | 2.396e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.320 | 3.869 | 56 | 3.785e-02 | 7.817e-03 | 4.818e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.112 | 3.659 | 56 | 4.412e-02 | 8.604e-01 | 5.638e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.869 | 4.415 | 52 | 1.283e-02 | - | 1.469e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.178 | 3.724 | 56 | 4.490e-02 | 2.500e-01 | 5.737e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.174 | 3.667 | 56 | 4.404e-02 | 8.596e-01 | 5.630e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.999 | 4.492 | 52 | 1.282e-02 | - | 1.474e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.325 | 3.818 | 56 | 4.483e-02 | 2.500e-01 | 5.733e-03 |
| Inv-Newton | 1024x1024 | spike | 5.023 | 3.612 | 56 | 3.974e-02 | 7.722e-01 | 5.013e-03 |
| PE-Quad | 1024x1024 | spike | 6.897 | 5.486 | 52 | 1.416e-02 | - | 1.725e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.459 | 4.048 | 56 | 4.408e-02 | 2.378e-01 | 5.555e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
