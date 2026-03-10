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
invM_std, info = torch.linalg.solve_ex(M_std, I)

alpha = b/c
beta = a - b/c
Q_std = alpha * I + beta * invM_std
X_next_std = X @ Q_std
print("Standard min SV:", torch.linalg.svdvals(X_next_std)[-1].item())

# Scaled way
M_scaled = symmetrize(S + (1.0 / c) * I)
invM_scaled, info2 = torch.linalg.solve_ex(M_scaled, I)

Q_scaled = alpha * I + (beta / c) * invM_scaled
X_next = X @ Q_scaled
print("Scaled min SV:", torch.linalg.svdvals(X_next)[-1].item())

# True theoretical mapped value for 1e-6:
print("Theoretical mapped value:", (a + b*1e-12)/(1 + c*1e-12) * 1e-6)
