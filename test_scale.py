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

# Standard way
I = torch.eye(n, dtype=torch.float32)
M_std = symmetrize(I + c * S)
try:
    invM_std, info = torch.linalg.solve_ex(M_std, I)
    print("Standard solve_ex info:", info.item())
except Exception as e:
    print("Standard failed", e)

# Scaled way
M_scaled = symmetrize(S + (1.0 / c) * I)
invM_scaled, info2 = torch.linalg.solve_ex(M_scaled, I)
print("Scaled solve_ex info:", info2.item())

Q_scaled = (b/c) * I + ((a - b/c) / c) * invM_scaled
X_next = X @ Q_scaled

s_next = torch.linalg.svdvals(X_next)
print("New min SV:", s_next[-1].item())
print("New max SV:", s_next[0].item())
