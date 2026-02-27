from fast_iroot.coeffs import _quad_coeffs


def test_quad_coeffs_reuses_normalized_float_list():
    coeffs = [(1.5, -0.5, 0.0), (1.25, -0.3, 0.05)]
    out = _quad_coeffs(coeffs)
    assert out is coeffs


def test_quad_coeffs_casts_non_float_sequence_items():
    out = _quad_coeffs([(1, -1, 0), (2.0, -1.0, 0.5)])
    assert out == [(1.0, -1.0, 0.0), (2.0, -1.0, 0.5)]
    assert all(isinstance(x, float) for triple in out for x in triple)
