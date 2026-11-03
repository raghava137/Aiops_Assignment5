# Run from project root: PYTHONPATH=. pytest partB
from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 14.0, "p95_ms": 2.0}


def cand(**kw):
    c = {"mae": 9.0, "mae_rush": 11.0, "mae_rain": 13.0, "p95_ms": 3.0}
    c.update(kw)
    return c


def test_better_candidate_promotes():
    d = decide(PROD, cand())
    assert d["promote"] and all(d["criteria"].values())


def test_gain_just_inside():
    assert decide(PROD, cand(mae=9.69))["promote"]      # 3.1% gain


def test_gain_just_outside():
    d = decide(PROD, cand(mae=9.71))                     # 2.9% gain
    assert not d["promote"] and not d["criteria"]["mae_gain"]


def test_worse_candidate():
    assert not decide(PROD, cand(mae=10.5))["promote"]


def test_latency_boundary():
    assert decide(PROD, cand(p95_ms=9.99))["promote"]
    d = decide(PROD, cand(p95_ms=10.01))
    assert not d["promote"] and not d["criteria"]["p95_latency"]


def test_rush_boundary():                                # limit 12.6
    assert decide(PROD, cand(mae_rush=12.59))["promote"]
    d = decide(PROD, cand(mae_rush=12.61))
    assert not d["promote"] and not d["criteria"]["rush_slice"]


def test_rain_blocks_even_with_big_gain():               # limit 14.7
    assert decide(PROD, cand(mae_rain=14.69))["promote"]
    d = decide(PROD, cand(mae=5.0, mae_rain=14.71))
    assert d["criteria"]["mae_gain"] and not d["criteria"]["rain_slice"] and not d["promote"]


def test_custom_thresholds():
    assert not decide(PROD, cand(), min_rel_gain=0.60)["promote"]
    assert decide(PROD, cand(p95_ms=15), max_p95_ms=20)["promote"]
    assert decide(PROD, cand(mae_rush=13), max_slice_regression=0.10)["promote"]


def test_inputs_not_changed():
    p, c = dict(PROD), cand()
    decide(p, c)
    assert p == PROD and c == cand()
