import torch
n = 200
X = torch.randn(2000, n, dtype=torch.float64)
U, _, V = torch.svd(X, some=False)
s = torch.linspace(1e-6, 1.0, n, dtype=torch.float64)
X_exact = U[:, :n] @ torch.diag(s) @ V.T

X_fp32 = X_exact.to(torch.float32)
print("Min SV of X_fp32:", torch.linalg.svdvals(X_fp32)[-1].item())
