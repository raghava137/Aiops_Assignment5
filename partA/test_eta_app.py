# Run from project root: pytest partA
import pytest
from fastapi.testclient import TestClient

import eta_app

client = TestClient(eta_app.app)
URL = "/v1/models/eta:predict"
ROW = {"distance_km": 12.5, "n_stops": 3, "hour": 8, "weekday": 2, "traffic_index": 0.6, "rain": 0, "weight_kg": 4.0}


def post(rows, **kw):
    return client.post(URL, json={"instances": rows}, **kw)


def test_valid_request():
    r = post([ROW])
    j = r.json()
    assert r.status_code == 200
    assert len(j["predictions"]) == 1
    assert j["predictions"][0] == round(j["predictions"][0], 1)
    assert j["model_version"] == eta_app.VERSION
    assert j["request_id"] == r.headers["X-Request-ID"]


def test_batch_order():
    a, b = {**ROW, "distance_km": 2}, {**ROW, "distance_km": 300}
    single = [post([a]).json()["predictions"][0], post([b]).json()["predictions"][0]]
    assert post([a, b]).json()["predictions"] == single
    assert post([b, a]).json()["predictions"] == single[::-1]


@pytest.mark.parametrize("bad, field", [
    ({**ROW, "hour": 24}, "hour"),
    ({k: v for k, v in ROW.items() if k != "weight_kg"}, "weight_kg"),
    ({**ROW, "traffic_index": "heavy"}, "traffic_index"),
    ({**ROW, "rain": 2}, "rain"),
    ({**ROW, "weight_kg": 0}, "weight_kg"),
])
def test_invalid_input(bad, field):
    r = post([bad])
    j = r.json()
    assert r.status_code == 422
    assert j["error"] == "validation_failed"
    assert any(field in d["field"] for d in j["details"])
    assert j["request_id"] == r.headers["X-Request-ID"]


def test_infeasible():
    r = post([{**ROW, "distance_km": 151, "n_stops": 21}])
    assert r.status_code == 422
    assert "infeasible" in r.text


def test_feasible_at_boundary():
    assert post([{**ROW, "distance_km": 150, "n_stops": 60}]).status_code == 200
    assert post([{**ROW, "distance_km": 500, "n_stops": 20}]).status_code == 200


def test_batch_limits():
    assert post([]).status_code == 422
    assert post([ROW] * 501).status_code == 422
    assert len(post([ROW] * 500).json()["predictions"]) == 500


def test_request_id_echo():
    r = post([ROW], headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated():
    ok = post([ROW])
    bad = post([{**ROW, "hour": -1}])
    for r in (ok, bad):
        assert len(r.headers["X-Request-ID"]) == 36
        assert r.json()["request_id"] == r.headers["X-Request-ID"]
    assert ok.headers["X-Request-ID"] != bad.headers["X-Request-ID"]
    assert post([{**ROW, "hour": -1}], headers={"X-Request-ID": "e1"}).json()["request_id"] == "e1"


def test_metrics():
    post([ROW])
    client.get("/metrics")
    text = client.get("/metrics").text
    assert 'eta_requests_total{path="/v1/models/eta:predict",status="200"}' in text
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text
    assert text.count("eta_request_latency_seconds_bucket") >= 5
    assert 'path="/metrics"' not in text


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok", "model_version": eta_app.VERSION}


def test_generic_500(monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError("secret detail")
    monkeypatch.setattr(eta_app, "model", Broken())
    r = post([ROW], headers={"X-Request-ID": "boom"})
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text
    assert r.headers["X-Request-ID"] == "boom"
    assert r.json()["request_id"] == "boom"
