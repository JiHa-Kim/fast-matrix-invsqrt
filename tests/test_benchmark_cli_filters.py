import pytest

from benchmarks.solve.bench_solve_core import matrix_solve_methods
from benchmarks.solve.matrix_solve import _parse_methods_csv as parse_spd_methods
from benchmarks.solve.matrix_solve_nonspd import METHODS as NONSPD_METHODS
from benchmarks.solve.matrix_solve_nonspd import _parse_methods_csv as parse_nonspd_methods


def test_parse_spd_methods_defaults_to_all():
    available = matrix_solve_methods(2)
    assert parse_spd_methods("", available) == available


def test_parse_spd_methods_preserves_order_and_dedups():
    available = matrix_solve_methods(2)
    selected = parse_spd_methods(
        "PE-Quad-Coupled-Apply,Inverse-Newton-Coupled-Apply,PE-Quad-Coupled-Apply",
        available,
    )
    assert selected == [
        "PE-Quad-Coupled-Apply",
        "Inverse-Newton-Coupled-Apply",
    ]


def test_parse_spd_methods_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown method\\(s\\)"):
        parse_spd_methods("Nope-Method", matrix_solve_methods(2))


def test_parse_nonspd_methods_defaults_to_all():
    assert parse_nonspd_methods("", NONSPD_METHODS) == list(NONSPD_METHODS)


def test_parse_nonspd_methods_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown method\\(s\\)"):
        parse_nonspd_methods("Nope-Method", NONSPD_METHODS)
