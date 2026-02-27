import pytest

from benchmarks.run_benchmarks import _parse_rows, _to_markdown_ab


def test_parse_rows_extracts_p_field():
    raw = """
== SPD Size 128x128 | RHS 128x1 | dtype=torch.float32 ==
precond=jacobi | l_target=0.05 | p=2 | ruiz_iters=2
-- case gaussian_spd --
Inverse-Newton-Coupled-Apply      1.234 ms (pre 0.100 + iter 1.134) | relerr vs true: 1.000e-03
""".strip()
    rows = _parse_rows(raw, "spd")
    assert len(rows) == 1
    assert rows[0][:6] == (
        "spd",
        2,
        128,
        1,
        "gaussian_spd",
        "Inverse-Newton-Coupled-Apply",
    )


def test_ab_compare_can_match_without_method():
    rows_a = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "Inverse-Newton-Coupled-Apply",
            1.0,
            0.9,
            1.0e-3,
        )
    ]
    rows_b = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "PE-Quad-Coupled-Apply",
            0.8,
            0.7,
            1.2e-3,
        )
    ]
    md = _to_markdown_ab(
        rows_a,
        rows_b,
        label_a="baseline",
        label_b="candidate",
        match_on_method=False,
    )
    assert "baseline_method" in md
    assert "candidate_method" in md
    assert "Inverse-Newton-Coupled-Apply" in md
    assert "PE-Quad-Coupled-Apply" in md


def test_ab_compare_match_on_method_requires_same_method():
    rows_a = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "Inverse-Newton-Coupled-Apply",
            1.0,
            0.9,
            1.0e-3,
        )
    ]
    rows_b = [
        (
            "spd",
            2,
            128,
            1,
            "gaussian_spd",
            "PE-Quad-Coupled-Apply",
            0.8,
            0.7,
            1.2e-3,
        )
    ]
    with pytest.raises(RuntimeError, match="no overlapping keys"):
        _to_markdown_ab(
            rows_a,
            rows_b,
            label_a="baseline",
            label_b="candidate",
            match_on_method=True,
        )
