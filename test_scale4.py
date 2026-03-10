import torch
from polar.rational.dwh import dwh_coeffs_from_ell
from polar.ops import symmetrize

n = 200
X = torch.randn(2000, n, dtype=torch.float64)
U, _, V = torch.svd(X, some=False)
s = torch.linspace(1e-6, 1.0, n, dtype=torch.float64)
X = U[:, :n] @ torch.diag(s) @ V.T

# We have X in exact FP64. 
# We want to run the step in FP32 as much as possible.
X_fp32 = X.to(torch.float32)

ell = 1e-6
a, b, c = dwh_coeffs_from_ell(ell)
alpha = b/c
beta = a - b/c
I_fp32 = torch.eye(n, dtype=torch.float32)
I_fp64 = torch.eye(n, dtype=torch.float64)

# Method 1: purely FP32
S_fp32 = X_fp32.T @ X_fp32
M_fp32 = symmetrize(S_fp32 + (1.0/c) * I_fp32)
invM_fp32, _ = torch.linalg.solve_ex(M_fp32, I_fp32)
Q_fp32 = alpha * I_fp32 + (beta/c) * invM_fp32
X_next_fp32 = X_fp32 @ Q_fp32
print("Pure FP32 min SV:", torch.linalg.svdvals(X_next_fp32)[-1].item())

# Method 2: S in FP64, add shift in FP64, then cast to FP32 for solve
S_fp64 = X.T @ X
M_fp64 = symmetrize(S_fp64 + (1.0/c) * I_fp64)
M_mixed = M_fp64.to(torch.float32)
invM_mixed, _ = torch.linalg.solve_ex(M_mixed, I_fp32)
Q_mixed = alpha * I_fp32 + (beta/c) * invM_mixed
X_next_mixed = X_fp32 @ Q_mixed
print("Mixed (FP64 S+shift, FP32 solve) min SV:", torch.linalg.svdvals(X_next_mixed)[-1].item())

# Method 3: S in FP32, but add shift carefully?
# Wait, user said "scaling down the numbers then back up"
# Let M = I + c S
M_scale = I_fp32 + c * S_fp32
try:
    invM_scale, _ = torch.linalg.solve_ex(M_scale, I_fp32)
    Q_scale = alpha * I_fp32 + beta * invM_scale
    X_next_scale = X_fp32 @ Q_scale
    print("Standard M=I+cS min SV:", torch.linalg.svdvals(X_next_scale)[-1].item())
except Exception as e:
    print("Standard failed")

print("Theoretical mapped value:", (a + b*1e-12)/(1 + c*1e-12) * 1e-6)
