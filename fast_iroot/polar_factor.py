import torch

# Polar Express degree-5 schedule (Algorithm 1): p(x) = a x + b x^3 + c x^5
_PE5 = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


# Polar factor computation with Turbo-Muon AOL preconditioning + Polar Express
@torch.no_grad()
def polar_express(
    G: torch.Tensor, steps: int = 5, norm: str = "aol", eps: float = 1e-12
) -> torch.Tensor:
    X = G

    if norm == "aol":
        Gram = X.transpose(-2, -1) @ X  # A = XX^T
        # s_i = 1/sqrt(sum_j |(X^T X)_{ij}|)
        s = torch.rsqrt(Gram.abs().sum(dim=-1).clamp_min(eps))
        X = X * s.unsqueeze(-2)  # biased target
    elif norm == "fro":
        X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)  # unbiased target
    else:
        raise ValueError("norm must be 'aol' or 'fro'")

    coeffs = _PE5[:steps] + [_PE5[-1]] * max(0, steps - len(_PE5))
    for a, b, c in coeffs:
        A = X @ X.transpose(-2, -1)  # A = XX^T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X
