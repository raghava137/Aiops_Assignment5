# Run from project root with MLFLOW_TRACKING_URI=sqlite:///mlflow.db
#   PYTHONPATH=. python partB/eta_mlflow.py register | gate | shadow
import os
import sys
import time

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

import eta_data

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
NAME = "eta-model"
FEATURES = eta_data.FEATURES


class EtaModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        return pd.DataFrame({"eta_minutes": np.round(self.model.predict(X), 1), "model_version": self.version})


def register():
    mlflow.set_experiment("eta-serving")
    X = pd.read_csv("data/eta_train.csv")[FEATURES].astype("float64")
    sig = infer_signature(X.head(5), pd.DataFrame({"eta_minutes": [30.0], "model_version": ["v1"]}))
    reqs = [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"joblib=={joblib.__version__}"]
    versions = {}
    for v in ["v1", "v2"]:
        with mlflow.start_run(run_name=f"eta-{v}"):
            info = mlflow.pyfunc.log_model(name="model", python_model=EtaModel(),
                                           artifacts={"bundle": f"data/eta_{v}.joblib"},
                                           signature=sig, input_example=X.head(2),
                                           pip_requirements=reqs, registered_model_name=NAME)
        versions[v] = info.registered_model_version
    client = MlflowClient()
    client.set_registered_model_alias(NAME, "prod", versions["v1"])
    client.set_registered_model_alias(NAME, "candidate", versions["v2"])
    aliases = client.get_registered_model(NAME).aliases
    print("version | aliases")
    for mv in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda m: int(m.version)):
        print(mv.version, "|", [a for a, ver in aliases.items() if str(ver) == str(mv.version)])


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    criteria = {
        "mae_gain": (prod["mae"] - cand["mae"]) / prod["mae"] >= min_rel_gain,
        "p95_latency": cand["p95_ms"] <= max_p95_ms,
        "rush_slice": cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression),
        "rain_slice": cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def promotion_gate(**thresholds):
    d = eta_data.make_data(2000, seed=99)
    target = [c for c in d.columns if c not in FEATURES][0]
    X, y = d[FEATURES].astype("float64"), d[target].to_numpy()
    rush = (d["hour"].between(7, 9) | d["hour"].between(16, 19)).to_numpy()
    rain = (d["rain"] == 1).to_numpy()
    rows = [X.iloc[[i]] for i in range(200)]
    metrics = {}
    for alias in ["prod", "candidate"]:
        m = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")
        err = np.abs(m.predict(X)["eta_minutes"].to_numpy() - y)
        m.predict(rows[0])  # warm-up
        lat = []
        for r in rows:
            t = time.perf_counter()
            m.predict(r)
            lat.append((time.perf_counter() - t) * 1000)
        metrics[alias] = {"mae": round(float(err.mean()), 3), "mae_rush": round(float(err[rush].mean()), 3),
                          "mae_rain": round(float(err[rain].mean()), 3),
                          "p95_ms": round(float(np.percentile(lat, 95)), 3)}
    result = decide(metrics["prod"], metrics["candidate"], **thresholds)
    result["metrics"] = metrics
    return result


def shadow(n=200):
    rows = pd.read_csv("data/eta_fresh.csv")[FEATURES].head(n).to_dict(orient="records")
    ports = [5101, 5102]
    sess = {p: requests.Session() for p in ports}
    preds, lat, ver = {p: [] for p in ports}, {p: [] for p in ports}, {}
    for row in rows:
        for p in ports:
            t = time.perf_counter()
            r = sess[p].post(f"http://127.0.0.1:{p}/invocations", json={"dataframe_records": [row]}, timeout=10)
            lat[p].append((time.perf_counter() - t) * 1000)
            out = r.json()["predictions"][0]
            preds[p].append(out["eta_minutes"])
            ver[p] = out["model_version"]
    diff = np.abs(np.array(preds[5101]) - np.array(preds[5102]))
    print(f"{n} identical one-row requests; 5101 served {ver[5101]}, 5102 served {ver[5102]}")
    print(f"mean abs difference: {diff.mean():.2f} min")
    print(f"differ by > 10 min: {(diff > 10).mean():.1%}")
    print(f"p95 latency: prod {np.percentile(lat[5101], 95):.1f} ms, candidate {np.percentile(lat[5102], 95):.1f} ms")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "register"
    if cmd == "register":
        register()
    elif cmd == "gate":
        print("default:", promotion_gate())
        print("min_rel_gain=0.60:", promotion_gate(min_rel_gain=0.60))
    elif cmd == "shadow":
        shadow()
