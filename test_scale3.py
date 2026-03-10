import torch
from polar.rational.dwh import dwh_coeffs_from_ell
from polar.ops import symmetrize

n = 200
X = torch.randn(2000, n, dtype=torch.float32)
U, _, V = torch.svd(X, some=False)
s = torch.linspace(1e-6, 1.0, n)
X = U[:, :n] @ torch.diag(s) @ V.T

S = X.T @ X
ell = 1e-6
a, b, c = dwh_coeffs_from_ell(ell)

alpha = b/c
beta = a - b/c
I = torch.eye(n, dtype=torch.float32)

best_s = 1.0
best_sv = 0.0

for power in range(0, 8):
    scale = 10**power
    M = symmetrize((1.0/scale) * I + (c/scale) * S)
    try:
        invM, info = torch.linalg.solve_ex(M, I)
        Q = alpha * I + (beta/scale) * invM
        X_next = X @ Q
        sv = torch.linalg.svdvals(X_next)[-1].item()
        print(f"Scale 10^{power}: min SV = {sv}")
    except Exception as e:
        print(f"Scale 10^{power}: failed")

print("Theoretical mapped value:", (a + b*1e-12)/(1 + c*1e-12) * 1e-6)
