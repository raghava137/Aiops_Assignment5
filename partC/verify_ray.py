# Run from project root: PYTHONPATH=partC python partC/verify_ray.py | tee partC/verify_output.txt
import ray
import requests
from ray import serve

import eta_ray

URL = "http://127.0.0.1:8100/v1/models/eta:predict"
ROW = {"distance_km": 12.5, "n_stops": 3, "hour": 8, "weekday": 2, "traffic_index": 0.6, "rain": 0, "weight_kg": 4.0}
IDS = [f"customer-{i:04d}" for i in range(1000)]
sess = requests.Session()


def get_arms(ids, headers=None):
    arms = {}
    for cid in ids:
        r = sess.post(URL, json={"customer_id": cid, "instances": [ROW]}, headers=headers, timeout=30)
        assert r.status_code == 200, r.text
        assert r.json()["arm"] == r.json()["model_version"]
        arms[cid] = r.json()["arm"]
    return arms


def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name="eta", route_prefix="/")


ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR", log_to_driver=False)
serve.start(http_options={"host": "127.0.0.1", "port": 8100})
print("Ray CPUs:", ray.cluster_resources()["CPU"])

deploy(0.2)
first = get_arms(IDS)
share = list(first.values()).count("v2") / len(IDS)
print(f"(a) weight 0.2: {share:.1%} of 1000 customers on v2")
assert abs(share - 0.2) <= 0.04

second = get_arms(IDS[:200])
same = sum(second[c] == first[c] for c in second)
print(f"(b) second call: {same}/200 same arm")
assert same == 200

serve.delete("eta")
deploy(0.2)
third = get_arms(IDS[:200])
same = sum(third[c] == first[c] for c in third)
print(f"(c) after redeploy: {same}/200 same arm")
assert same == 200

deploy(0.5)
raised = get_arms(IDS)
moved_back = [c for c in IDS if first[c] == "v2" and raised[c] == "v1"]
share = list(raised.values()).count("v2") / len(IDS)
print(f"(d) weight 0.5: {share:.1%} on v2, moved from v2 back to v1: {len(moved_back)}")
assert not moved_back

prod_ids = [c for c in IDS if raised[c] == "v1"][:50]
forced = get_arms(prod_ids, headers={"X-Beta": "1"})
n = list(forced.values()).count("v2")
print(f"(e) X-Beta: 1 on 50 prod customers: {n}/50 on v2")
assert n == 50

deps = serve.status().applications["eta"].deployments
print("replicas:", {name: dict(d.replica_states) for name, d in deps.items()})
print("all checks passed")
serve.shutdown()
ray.shutdown()
