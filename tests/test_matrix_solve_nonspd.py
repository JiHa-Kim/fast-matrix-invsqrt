import pytest
import torch

from fast_iroot.nonspd import precond_nonspd


@pytest.mark.parametrize("mode", ["row-norm", "frob", "ruiz"])
def test_precond_nonspd_modes_finite(mode: str):
    torch.manual_seed(11)
    A = torch.randn(12, 12) + 0.25 * torch.eye(12)
    A_norm = precond_nonspd(A, mode=mode, ruiz_iters=2)
    assert A_norm.shape == A.shape
    assert torch.isfinite(A_norm).all()


def test_precond_nonspd_rejects_invalid_mode():
    A = torch.eye(4)
    with pytest.raises(ValueError, match="Unknown non-SPD preconditioner mode"):
        precond_nonspd(A, mode="bad-mode")


def test_precond_nonspd_ruiz_iters_validation():
    A = torch.eye(4)
    with pytest.raises(ValueError, match="ruiz_iters"):
        precond_nonspd(A, mode="ruiz", ruiz_iters=0)
