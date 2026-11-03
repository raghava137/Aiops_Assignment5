# python partD/smoke_eta.py --url http://localhost:8000 --expect-version v1 [--max-p95-ms 80]
import argparse
import math
import sys
import time

import numpy as np
import requests

ROW = {"distance_km": 20.0, "n_stops": 4, "hour": 11, "weekday": 2, "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--expect-version", required=True)
ap.add_argument("--max-p95-ms", type=float, default=80.0)
args = ap.parse_args()
BASE = args.url.rstrip("/")
sess = requests.Session()


def predict(rows):
    r = sess.post(BASE + "/v1/models/eta:predict", json={"customer_id": "smoke", "instances": rows}, timeout=10)
    return r, (r.json() if r.status_code == 200 else {})


def health():
    r = sess.get(BASE + "/healthz", timeout=5)
    return r.status_code == 200 and r.json().get("status") == "ok", f"HTTP {r.status_code}"


def version_and_value():
    r, j = predict([ROW])
    p = j.get("predictions", [None])[0]
    ok = j.get("model_version") == args.expect_version and p is not None and math.isfinite(p) and 0 < p < 1440
    return ok, f"version={j.get('model_version')} eta={p}"


def batch_order():
    rows = [{**ROW, "distance_km": d} for d in (2, 80, 30)]
    batch = predict(rows)[1].get("predictions")
    single = [predict([x])[1].get("predictions", [None])[0] for x in rows]
    return batch == single, f"batch={batch} single={single}"


def longer_distance_not_faster():
    p = predict([{**ROW, "distance_km": 5}, {**ROW, "distance_km": 50}])[1].get("predictions", [0, -1])
    return p[1] >= p[0], f"5 km={p[0]} 50 km={p[1]}"


def more_traffic_not_faster():
    p = predict([{**ROW, "traffic_index": 0.1}, {**ROW, "traffic_index": 0.9}])[1].get("predictions", [0, -1])
    return p[1] >= p[0], f"traffic 0.1={p[0]} 0.9={p[1]}"


def invalid_gives_422():
    r, _ = predict([{**ROW, "distance_km": -5}])
    return r.status_code == 422, f"HTTP {r.status_code}"


def p95_latency():
    lat = []
    for _ in range(50):
        t = time.perf_counter()
        predict([ROW])
        lat.append((time.perf_counter() - t) * 1000)
    p95 = np.percentile(lat, 95)
    return p95 <= args.max_p95_ms, f"p95={p95:.1f} ms"


failed = 0
for check in [health, version_and_value, batch_order, longer_distance_not_faster,
              more_traffic_not_faster, invalid_gives_422, p95_latency]:
    try:
        ok, info = check()
    except Exception as e:
        ok, info = False, str(e)
    failed += not ok
    print("PASS" if ok else "FAIL", check.__name__, info)
print("smoke test", "passed" if failed == 0 else f"failed ({failed})")
sys.exit(1 if failed else 0)
