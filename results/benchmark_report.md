# Fast Matrix Inverse p-th Roots Benchmark Report
*Date: 2026-02-24*

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
| Inv-Newton | 256x256 | gaussian_spd | 1143.055 | 1141.635 | 11 | 2.362e-03 | 1.261e-01 | 2.363e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.379 | 0.959 | 11 | 5.079e-03 | - | 5.072e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.709 | 1.289 | 11 | 7.439e-03 | 2.513e-01 | 7.414e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.548 | 1.116 | 11 | 1.573e-03 | 1.256e-01 | 1.560e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.384 | 0.952 | 11 | 7.797e-03 | - | 7.801e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.739 | 1.307 | 11 | 6.631e-03 | 1.534e-01 | 6.617e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.890 | 1.127 | 11 | 1.090e-03 | 1.253e-01 | 1.083e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.729 | 0.967 | 11 | 7.945e-03 | - | 7.948e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.701 | 0.938 | 11 | 7.222e-03 | 9.363e-02 | 7.210e-03 |
| Inv-Newton | 256x256 | near_rank_def | 3.173 | 1.196 | 11 | 2.213e-03 | 1.253e-01 | 2.222e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.075 | 1.098 | 11 | 8.957e-03 | - | 8.963e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.936 | 0.958 | 11 | 8.685e-03 | 6.010e-02 | 8.660e-03 |
| Inv-Newton | 256x256 | spike | 2.596 | 1.139 | 11 | 5.061e-03 | 1.237e-01 | 5.069e-03 |
| PE-Quad | 256x256 | spike | 2.475 | 1.018 | 11 | 6.774e-03 | - | 6.780e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.503 | 1.046 | 11 | 6.863e-03 | 1.195e-01 | 6.856e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.457 | 1.072 | 20 | 4.163e-03 | 1.776e-01 | 4.135e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.613 | 1.228 | 19 | 3.649e-03 | - | 3.657e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.453 | 1.068 | 20 | 2.233e-03 | 1.912e-01 | 2.235e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.820 | 1.059 | 20 | 2.296e-03 | 1.773e-01 | 2.275e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.918 | 1.157 | 19 | 8.464e-03 | - | 8.478e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.786 | 1.025 | 20 | 5.794e-03 | 5.811e-02 | 5.790e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.754 | 1.248 | 20 | 3.399e-03 | 1.771e-01 | 3.383e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.646 | 1.141 | 19 | 4.473e-03 | - | 4.491e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 2.793 | 1.287 | 20 | 4.638e-03 | 9.510e-02 | 4.647e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.800 | 1.102 | 20 | 1.571e-03 | 1.771e-01 | 1.555e-03 |
| PE-Quad | 512x512 | near_rank_def | 4.213 | 2.515 | 19 | 7.056e-03 | - | 7.063e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 4.225 | 2.527 | 20 | 7.313e-03 | 1.458e-01 | 7.302e-03 |
| Inv-Newton | 512x512 | spike | 2.623 | 1.035 | 20 | 2.308e-03 | 1.733e-01 | 2.294e-03 |
| PE-Quad | 512x512 | spike | 2.548 | 0.960 | 19 | 6.328e-03 | - | 6.337e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.523 | 0.935 | 20 | 6.328e-03 | 1.730e-01 | 6.337e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 3.648 | 2.221 | 56 | 3.423e-03 | 2.509e-01 | 3.401e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 3.984 | 2.558 | 52 | 4.964e-03 | - | 4.985e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 3.765 | 2.339 | 56 | 4.971e-03 | 1.756e-01 | 4.988e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.213 | 2.345 | 56 | 4.391e-03 | 2.505e-01 | 4.378e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.410 | 2.543 | 52 | 3.683e-04 | - | 3.610e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.485 | 2.617 | 56 | 4.123e-04 | 3.546e-02 | 4.165e-04 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 3.753 | 2.316 | 56 | 1.109e-03 | 2.504e-01 | 1.097e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 3.919 | 2.482 | 52 | 7.101e-03 | - | 7.106e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 3.654 | 2.217 | 56 | 7.101e-03 | 2.533e-01 | 7.108e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 3.805 | 2.362 | 56 | 1.181e-03 | 2.503e-01 | 1.169e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 3.970 | 2.527 | 52 | 7.034e-03 | - | 7.039e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 3.800 | 2.357 | 56 | 7.034e-03 | 2.529e-01 | 7.040e-03 |
| Inv-Newton | 1024x1024 | spike | 3.757 | 2.248 | 56 | 2.395e-03 | 2.460e-01 | 2.376e-03 |
| PE-Quad | 1024x1024 | spike | 4.042 | 2.533 | 52 | 6.430e-03 | - | 6.432e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 3.921 | 2.412 | 56 | 6.126e-03 | 2.401e-01 | 6.132e-03 |


## Results for $p=2$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 2913.230 | 2911.248 | 11 | 4.411e-03 | 6.916e-02 | 2.183e-03 |
| PE-Quad | 256x256 | gaussian_spd | 3.268 | 1.286 | 11 | 4.288e-03 | - | 2.134e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 3.322 | 1.340 | 11 | 4.297e-03 | 4.539e-02 | 2.139e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 5.835 | 1.603 | 11 | 3.590e-03 | 6.262e-02 | 1.793e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 5.516 | 1.283 | 11 | 3.645e-03 | - | 1.817e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 5.207 | 0.974 | 11 | 3.657e-03 | 3.945e-02 | 1.824e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.613 | 1.225 | 11 | 2.231e-03 | 3.917e-02 | 1.119e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.630 | 1.241 | 11 | 2.282e-03 | - | 1.141e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.412 | 1.023 | 11 | 2.543e-03 | 2.376e-02 | 1.259e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.824 | 1.268 | 11 | 1.662e-03 | 8.211e-03 | 8.359e-04 |
| PE-Quad | 256x256 | near_rank_def | 2.992 | 1.436 | 11 | 3.513e-03 | - | 1.763e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.833 | 1.277 | 11 | 2.271e-03 | 3.907e-03 | 1.129e-03 |
| Inv-Newton | 256x256 | spike | 3.270 | 1.178 | 11 | 3.466e-03 | 7.695e-02 | 1.784e-03 |
| PE-Quad | 256x256 | spike | 3.565 | 1.474 | 11 | 5.674e-03 | - | 2.878e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.442 | 1.350 | 11 | 1.160e-02 | 5.593e-02 | 5.846e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.794 | 1.287 | 20 | 4.246e-03 | 1.600e-01 | 2.084e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.746 | 1.239 | 19 | 4.251e-03 | - | 2.080e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.621 | 1.114 | 20 | 6.033e-03 | 8.839e-02 | 2.981e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.676 | 1.131 | 20 | 2.667e-03 | 1.133e-01 | 1.382e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 2.666 | 1.121 | 19 | 2.668e-03 | - | 1.380e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.849 | 1.304 | 20 | 2.884e-03 | 5.661e-02 | 1.491e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.533 | 0.991 | 20 | 3.596e-03 | 1.575e-01 | 1.834e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 2.854 | 1.313 | 19 | 3.594e-03 | - | 1.833e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.241 | 1.700 | 20 | 8.799e-03 | 7.564e-02 | 4.411e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.448 | 1.030 | 20 | 1.602e-03 | 6.819e-02 | 8.072e-04 |
| PE-Quad | 512x512 | near_rank_def | 2.506 | 1.087 | 19 | 1.601e-03 | - | 8.057e-04 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.852 | 1.434 | 20 | 1.040e-02 | 3.125e-02 | 5.146e-03 |
| Inv-Newton | 512x512 | spike | 2.651 | 1.102 | 20 | 1.363e-03 | 6.287e-02 | 6.953e-04 |
| PE-Quad | 512x512 | spike | 2.788 | 1.239 | 19 | 1.681e-03 | - | 8.307e-04 |
| PE-Quad-Coupled | 512x512 | spike | 2.652 | 1.103 | 20 | 1.113e-02 | 7.182e-02 | 5.651e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.058 | 2.673 | 56 | 3.198e-03 | 2.030e-01 | 1.546e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 4.520 | 3.134 | 52 | 3.196e-03 | - | 1.544e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.074 | 2.688 | 56 | 1.111e-02 | 2.750e-01 | 5.587e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.295 | 2.693 | 56 | 3.885e-03 | 2.501e-01 | 1.842e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 4.724 | 3.122 | 52 | 3.884e-03 | - | 1.840e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.272 | 2.671 | 56 | 1.088e-02 | 2.795e-01 | 5.469e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.345 | 2.797 | 56 | 4.697e-04 | 4.247e-03 | 1.584e-04 |
| PE-Quad | 1024x1024 | illcond_1e12 | 4.636 | 3.088 | 52 | 4.681e-04 | - | 1.552e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.322 | 2.774 | 56 | 1.154e-02 | 2.652e-01 | 5.794e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.212 | 2.719 | 56 | 4.867e-04 | 3.868e-03 | 1.833e-04 |
| PE-Quad | 1024x1024 | near_rank_def | 4.520 | 3.027 | 52 | 4.853e-04 | - | 1.807e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.235 | 2.741 | 56 | 1.155e-02 | 2.655e-01 | 5.816e-03 |
| Inv-Newton | 1024x1024 | spike | 4.735 | 2.751 | 56 | 1.389e-03 | 8.752e-02 | 7.195e-04 |
| PE-Quad | 1024x1024 | spike | 5.137 | 3.154 | 52 | 1.710e-03 | - | 8.645e-04 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.827 | 2.844 | 56 | 1.033e-02 | 2.119e-01 | 5.223e-03 |


## Results for $p=3$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 1268.451 | 1266.928 | 11 | 1.702e-02 | 1.285e-01 | 5.704e-03 |
| PE-Quad | 256x256 | gaussian_spd | 5.345 | 3.822 | 11 | 6.827e-03 | - | 2.233e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 4.321 | 2.798 | 11 | 6.351e-03 | 1.215e-01 | 2.068e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.717 | 1.161 | 11 | 1.592e-02 | 5.597e-02 | 5.410e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 3.163 | 1.607 | 11 | 5.571e-03 | - | 1.818e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 3.659 | 2.103 | 11 | 5.581e-03 | 5.003e-02 | 1.817e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.960 | 1.170 | 11 | 1.607e-02 | 5.836e-02 | 5.490e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 3.116 | 1.327 | 11 | 3.675e-03 | - | 1.216e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.965 | 1.176 | 11 | 3.669e-03 | 2.067e-02 | 1.215e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.881 | 1.314 | 11 | 1.580e-02 | 5.861e-02 | 5.396e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.848 | 1.281 | 11 | 2.881e-03 | - | 9.590e-04 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.653 | 1.086 | 11 | 2.873e-03 | 5.469e-02 | 9.601e-04 |
| Inv-Newton | 256x256 | spike | 2.809 | 1.165 | 11 | 1.383e-02 | 5.807e-02 | 4.600e-03 |
| PE-Quad | 256x256 | spike | 2.917 | 1.274 | 11 | 5.922e-03 | - | 1.956e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.725 | 1.081 | 11 | 5.961e-03 | 1.036e-01 | 2.014e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 3.413 | 1.495 | 20 | 1.357e-02 | 1.402e-01 | 4.615e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.189 | 1.271 | 19 | 8.307e-03 | - | 2.898e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.069 | 1.151 | 20 | 8.373e-03 | 9.343e-02 | 2.827e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.904 | 1.114 | 20 | 1.493e-02 | 7.539e-02 | 5.003e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 4.573 | 2.783 | 19 | 5.935e-03 | - | 2.073e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 3.649 | 1.859 | 20 | 6.068e-03 | 4.803e-04 | 2.117e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.514 | 1.393 | 20 | 1.246e-02 | 5.244e-02 | 4.083e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.602 | 1.481 | 19 | 8.417e-03 | - | 2.755e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.722 | 1.601 | 20 | 8.476e-03 | 4.025e-04 | 2.910e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.853 | 1.368 | 20 | 1.601e-02 | 8.325e-02 | 5.263e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.006 | 1.522 | 19 | 4.134e-03 | - | 1.399e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.054 | 1.570 | 20 | 4.139e-03 | 3.897e-04 | 1.447e-03 |
| Inv-Newton | 512x512 | spike | 2.726 | 1.237 | 20 | 1.517e-02 | 1.044e-01 | 5.064e-03 |
| PE-Quad | 512x512 | spike | 2.808 | 1.320 | 19 | 4.088e-03 | - | 1.352e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.688 | 1.199 | 20 | 4.109e-03 | 6.201e-02 | 1.362e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.850 | 3.272 | 56 | 1.307e-02 | 8.469e-02 | 4.282e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.385 | 3.807 | 52 | 7.602e-03 | - | 2.462e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.920 | 3.342 | 56 | 7.848e-03 | 1.455e-01 | 2.606e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.640 | 3.196 | 56 | 9.036e-03 | 4.737e-03 | 2.937e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.184 | 3.740 | 52 | 9.032e-03 | - | 2.937e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.671 | 3.227 | 56 | 9.525e-03 | 1.854e-01 | 3.157e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.653 | 3.253 | 56 | 1.630e-02 | 1.250e-01 | 5.388e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.224 | 3.825 | 52 | 2.632e-03 | - | 9.131e-04 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.582 | 3.183 | 56 | 2.634e-03 | 5.389e-04 | 9.134e-04 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.584 | 3.161 | 56 | 1.623e-02 | 1.265e-01 | 5.373e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.138 | 3.715 | 52 | 2.706e-03 | - | 9.293e-04 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.597 | 3.173 | 56 | 2.707e-03 | 5.113e-04 | 9.296e-04 |
| Inv-Newton | 1024x1024 | spike | 4.574 | 3.146 | 56 | 1.498e-02 | 1.549e-01 | 5.022e-03 |
| PE-Quad | 1024x1024 | spike | 5.210 | 3.783 | 52 | 4.217e-03 | - | 1.388e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.593 | 3.165 | 56 | 4.174e-03 | 8.594e-02 | 1.372e-03 |


## Results for $p=4$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 1194.438 | 1193.049 | 11 | 7.606e-03 | 1.561e-01 | 2.014e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.660 | 1.271 | 11 | 7.606e-03 | - | 2.014e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.777 | 1.387 | 11 | 1.548e-02 | 1.013e-01 | 3.923e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 2.642 | 1.143 | 11 | 1.080e-02 | 1.530e-01 | 2.800e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.887 | 1.388 | 11 | 1.157e-02 | - | 2.936e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.873 | 1.373 | 11 | 1.364e-02 | 7.534e-02 | 3.370e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.820 | 1.246 | 11 | 1.138e-02 | 1.357e-01 | 2.917e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.880 | 1.307 | 11 | 1.312e-02 | - | 3.212e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.650 | 1.076 | 11 | 1.417e-02 | 4.688e-02 | 3.427e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.697 | 1.157 | 11 | 1.133e-02 | 1.197e-01 | 2.899e-03 |
| PE-Quad | 256x256 | near_rank_def | 2.876 | 1.335 | 11 | 1.321e-02 | - | 3.213e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.889 | 1.349 | 11 | 1.407e-02 | 1.353e-02 | 3.394e-03 |
| Inv-Newton | 256x256 | spike | 3.240 | 1.189 | 11 | 1.102e-02 | 1.506e-01 | 2.772e-03 |
| PE-Quad | 256x256 | spike | 3.353 | 1.302 | 11 | 1.134e-02 | - | 2.780e-03 |
| PE-Quad-Coupled | 256x256 | spike | 3.325 | 1.274 | 11 | 1.130e-02 | 7.694e-02 | 2.769e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.928 | 1.452 | 20 | 8.510e-03 | 2.513e-01 | 2.019e-03 |
| PE-Quad | 512x512 | gaussian_spd | 2.912 | 1.436 | 19 | 9.241e-03 | - | 2.149e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 2.646 | 1.171 | 20 | 9.214e-03 | 1.597e-01 | 2.149e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 2.863 | 1.193 | 20 | 1.141e-02 | 2.174e-01 | 2.838e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 3.094 | 1.423 | 19 | 1.288e-02 | - | 3.080e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 2.806 | 1.136 | 20 | 1.289e-02 | 1.132e-01 | 3.080e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 3.044 | 1.345 | 20 | 9.496e-03 | 2.495e-01 | 2.305e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.427 | 1.728 | 19 | 1.010e-02 | - | 2.407e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.137 | 1.438 | 20 | 1.011e-02 | 1.574e-01 | 2.408e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.782 | 1.307 | 20 | 1.233e-02 | 1.925e-01 | 3.083e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.099 | 1.624 | 19 | 1.351e-02 | - | 3.268e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 3.090 | 1.615 | 20 | 1.352e-02 | 6.811e-02 | 3.268e-03 |
| Inv-Newton | 512x512 | spike | 2.621 | 1.157 | 20 | 1.259e-02 | 1.884e-01 | 3.184e-03 |
| PE-Quad | 512x512 | spike | 3.031 | 1.568 | 19 | 1.263e-02 | - | 3.124e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.529 | 1.065 | 20 | 1.276e-02 | 7.625e-02 | 3.153e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 4.750 | 3.089 | 56 | 1.010e-02 | 3.375e-01 | 2.477e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.577 | 3.916 | 52 | 1.063e-02 | - | 2.559e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 4.741 | 3.080 | 56 | 1.039e-02 | 2.028e-01 | 2.520e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 4.858 | 3.203 | 56 | 6.457e-03 | 3.750e-01 | 1.514e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 5.486 | 3.831 | 52 | 6.460e-03 | - | 1.514e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 4.836 | 3.181 | 56 | 6.464e-03 | 2.500e-01 | 1.514e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 4.917 | 3.163 | 56 | 1.285e-02 | 2.500e-01 | 3.291e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 5.538 | 3.784 | 52 | 1.374e-02 | - | 3.357e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 4.967 | 3.212 | 56 | 1.330e-02 | 3.611e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 4.475 | 3.101 | 56 | 1.292e-02 | 2.500e-01 | 3.303e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.076 | 3.702 | 52 | 1.367e-02 | - | 3.346e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 4.497 | 3.122 | 56 | 1.329e-02 | 3.429e-04 | 3.322e-03 |
| Inv-Newton | 1024x1024 | spike | 4.959 | 3.188 | 56 | 1.287e-02 | 2.662e-01 | 3.235e-03 |
| PE-Quad | 1024x1024 | spike | 5.554 | 3.783 | 52 | 1.269e-02 | - | 3.169e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 4.849 | 3.078 | 56 | 1.290e-02 | 1.967e-01 | 3.220e-03 |


## Results for $p=8$

| Method | Size | Case | Total Time (ms) | Iter Time (ms) | Mem (MB) | Residual (med) | Y_res | Rel Err |
|---|---|---|---|---|---|---|---|---|
| Inv-Newton | 256x256 | gaussian_spd | 1131.954 | 1130.612 | 11 | 5.082e-02 | 4.339e-01 | 6.377e-03 |
| PE-Quad | 256x256 | gaussian_spd | 2.868 | 1.527 | 11 | 2.674e-02 | - | 3.255e-03 |
| PE-Quad-Coupled | 256x256 | gaussian_spd | 2.555 | 1.213 | 11 | 5.068e-02 | 2.335e-01 | 6.363e-03 |
| Inv-Newton | 256x256 | illcond_1e6 | 3.286 | 1.688 | 11 | 4.548e-02 | 4.416e-01 | 5.718e-03 |
| PE-Quad | 256x256 | illcond_1e6 | 2.931 | 1.332 | 11 | 1.437e-02 | - | 1.730e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e6 | 2.768 | 1.169 | 11 | 4.548e-02 | 2.310e-01 | 5.718e-03 |
| Inv-Newton | 256x256 | illcond_1e12 | 2.699 | 1.248 | 11 | 4.552e-02 | 4.415e-01 | 5.743e-03 |
| PE-Quad | 256x256 | illcond_1e12 | 2.907 | 1.456 | 11 | 1.257e-02 | - | 1.510e-03 |
| PE-Quad-Coupled | 256x256 | illcond_1e12 | 2.705 | 1.254 | 11 | 4.552e-02 | 2.368e-01 | 5.743e-03 |
| Inv-Newton | 256x256 | near_rank_def | 2.796 | 1.343 | 11 | 4.748e-02 | 4.394e-01 | 6.017e-03 |
| PE-Quad | 256x256 | near_rank_def | 3.106 | 1.654 | 11 | 1.089e-02 | - | 1.294e-03 |
| PE-Quad-Coupled | 256x256 | near_rank_def | 2.700 | 1.248 | 11 | 4.748e-02 | 2.493e-01 | 6.017e-03 |
| Inv-Newton | 256x256 | spike | 2.592 | 1.211 | 11 | 4.770e-02 | 3.999e-01 | 6.101e-03 |
| PE-Quad | 256x256 | spike | 2.831 | 1.451 | 11 | 1.618e-02 | - | 1.931e-03 |
| PE-Quad-Coupled | 256x256 | spike | 2.578 | 1.198 | 11 | 4.951e-02 | 2.292e-01 | 6.331e-03 |
| Inv-Newton | 512x512 | gaussian_spd | 2.662 | 1.232 | 20 | 4.031e-02 | 5.465e-01 | 5.075e-03 |
| PE-Quad | 512x512 | gaussian_spd | 3.589 | 2.159 | 19 | 2.340e-02 | - | 2.876e-03 |
| PE-Quad-Coupled | 512x512 | gaussian_spd | 3.104 | 1.674 | 20 | 4.031e-02 | 2.190e-01 | 5.075e-03 |
| Inv-Newton | 512x512 | illcond_1e6 | 4.971 | 2.179 | 20 | 4.377e-02 | 5.959e-01 | 5.553e-03 |
| PE-Quad | 512x512 | illcond_1e6 | 4.214 | 1.422 | 19 | 1.553e-02 | - | 1.846e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e6 | 4.281 | 1.489 | 20 | 4.377e-02 | 2.901e-01 | 5.553e-03 |
| Inv-Newton | 512x512 | illcond_1e12 | 2.860 | 1.222 | 20 | 4.088e-02 | 5.623e-01 | 5.190e-03 |
| PE-Quad | 512x512 | illcond_1e12 | 3.205 | 1.567 | 19 | 1.845e-02 | - | 2.193e-03 |
| PE-Quad-Coupled | 512x512 | illcond_1e12 | 3.589 | 1.951 | 20 | 4.093e-02 | 1.996e-01 | 5.196e-03 |
| Inv-Newton | 512x512 | near_rank_def | 2.997 | 1.435 | 20 | 4.432e-02 | 6.060e-01 | 5.642e-03 |
| PE-Quad | 512x512 | near_rank_def | 3.075 | 1.513 | 19 | 1.371e-02 | - | 1.602e-03 |
| PE-Quad-Coupled | 512x512 | near_rank_def | 2.928 | 1.366 | 20 | 4.455e-02 | 2.534e-01 | 5.672e-03 |
| Inv-Newton | 512x512 | spike | 2.748 | 1.296 | 20 | 4.168e-02 | 5.770e-01 | 5.303e-03 |
| PE-Quad | 512x512 | spike | 2.937 | 1.486 | 19 | 1.419e-02 | - | 1.718e-03 |
| PE-Quad-Coupled | 512x512 | spike | 2.692 | 1.240 | 20 | 4.396e-02 | 1.683e-01 | 5.589e-03 |
| Inv-Newton | 1024x1024 | gaussian_spd | 5.192 | 3.694 | 56 | 4.087e-02 | 7.994e-01 | 5.201e-03 |
| PE-Quad | 1024x1024 | gaussian_spd | 5.930 | 4.433 | 52 | 1.790e-02 | - | 2.110e-03 |
| PE-Quad-Coupled | 1024x1024 | gaussian_spd | 5.191 | 3.694 | 56 | 4.141e-02 | 2.500e-01 | 5.272e-03 |
| Inv-Newton | 1024x1024 | illcond_1e6 | 5.381 | 3.684 | 56 | 3.742e-02 | 7.435e-01 | 4.761e-03 |
| PE-Quad | 1024x1024 | illcond_1e6 | 6.079 | 4.383 | 52 | 2.033e-02 | - | 2.396e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e6 | 5.421 | 3.724 | 56 | 3.784e-02 | 2.500e-01 | 4.818e-03 |
| Inv-Newton | 1024x1024 | illcond_1e12 | 5.334 | 3.710 | 56 | 4.412e-02 | 8.590e-01 | 5.638e-03 |
| PE-Quad | 1024x1024 | illcond_1e12 | 6.047 | 4.424 | 52 | 1.283e-02 | - | 1.469e-03 |
| PE-Quad-Coupled | 1024x1024 | illcond_1e12 | 5.329 | 3.705 | 56 | 4.490e-02 | 2.500e-01 | 5.737e-03 |
| Inv-Newton | 1024x1024 | near_rank_def | 5.155 | 3.651 | 56 | 4.404e-02 | 8.582e-01 | 5.630e-03 |
| PE-Quad | 1024x1024 | near_rank_def | 5.940 | 4.436 | 52 | 1.282e-02 | - | 1.474e-03 |
| PE-Quad-Coupled | 1024x1024 | near_rank_def | 5.399 | 3.894 | 56 | 4.483e-02 | 2.500e-01 | 5.733e-03 |
| Inv-Newton | 1024x1024 | spike | 5.384 | 3.591 | 56 | 3.971e-02 | 7.744e-01 | 5.009e-03 |
| PE-Quad | 1024x1024 | spike | 6.180 | 4.387 | 52 | 1.416e-02 | - | 1.725e-03 |
| PE-Quad-Coupled | 1024x1024 | spike | 5.502 | 3.709 | 56 | 4.422e-02 | 2.451e-01 | 5.572e-03 |


## Summary
The benchmark results confirm the efficiency and robustness of the `PE-Quad` implementations across various condition numbers and exponents. The compiled GPU speeds demonstrate competitive execution profiles under real workloads.
