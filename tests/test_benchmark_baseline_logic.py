from benchmarks.run_benchmarks import RunSpec, _parse_csv_tokens, _filter_specs


def test_parse_csv_tokens():
    assert _parse_csv_tokens("a,b , c") == ["a", "b", "c"]
    assert _parse_csv_tokens("") == []
    assert _parse_csv_tokens(None) == []


def test_filter_specs_by_only():
    specs = [
        RunSpec("SPD p=1", "spd", [], "out1.txt", {}),
        RunSpec("Non-SPD p=1", "nonspd", [], "out2.txt", {}),
    ]
    assert len(_filter_specs(specs, ["non-spd"], [], [], [])) == 1
    assert _filter_specs(specs, ["non-spd"], [], [], [])[0].name == "Non-SPD p=1"


def test_filter_specs_by_kind():
    specs = [
        RunSpec("S1", "spd", [], "o1.txt", {"kind": "spd"}),
        RunSpec("G1", "spd", [], "o2.txt", {"kind": "gram"}),
    ]
    assert len(_filter_specs(specs, [], ["gram"], [], [])) == 1
    assert _filter_specs(specs, [], ["gram"], [], [])[0].tags["kind"] == "gram"


def test_filter_specs_by_p_val():
    specs = [
        RunSpec("P1", "spd", [], "o1.txt", {"p": "1"}),
        RunSpec("P2", "spd", [], "o2.txt", {"p": "2"}),
    ]
    assert len(_filter_specs(specs, [], [], ["2"], [])) == 1
    assert _filter_specs(specs, [], [], ["2"], [])[0].tags["p"] == "2"


def test_filter_specs_by_size():
    specs = [
        RunSpec("N256", "spd", [], "o1.txt", {"n": "256"}),
        RunSpec("N512", "spd", [], "o2.txt", {"n": "512"}),
        RunSpec("M256", "spd", [], "o3.txt", {"m": "256"}),
    ]
    # "256" should match both n=256 and m=256
    assert len(_filter_specs(specs, [], [], [], ["256"])) == 2
