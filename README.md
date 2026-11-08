# DA3408 Module 4 Assignment 5

Raghava R (DA24B021). The written answers and results are in `REPORT.pdf`.

## Folders

```
partA/  eta_app.py, test_eta_app.py, loadtest_results.json
partB/  eta_mlflow.py, test_gate.py, gate_output.txt, shadow_output.txt
partC/  eta_ray.py, verify_ray.py, verify_output.txt
partD/  smoke_eta.py
```

## Running it

Put the given `eta_data.py`, `load_test.py` and `requirements.txt` in this folder and run everything from here.

```bash
pip install -r requirements.txt prometheus_client
python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
```

Part A (start the server in one terminal, run the load tests in another):

```bash
pytest partA
ETA_MODEL=data/eta_v1.joblib uvicorn --app-dir partA eta_app:app --port 8000
python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results.json
python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100
```

For A3.4 I restarted the server with `--workers 2` and ran the first load test again.

Part B (`export MLFLOW_TRACKING_URI=sqlite:///mlflow.db` in every terminal used here):

```bash
PYTHONPATH=. pytest partB
PYTHONPATH=. python partB/eta_mlflow.py register
PYTHONPATH=. python partB/eta_mlflow.py gate > partB/gate_output.txt
mlflow models serve -m "models:/eta-model@prod" -p 5101 --env-manager local
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
PYTHONPATH=. python partB/eta_mlflow.py shadow > partB/shadow_output.txt
```

The two `mlflow models serve` commands go in separate terminals and must be running before `shadow`.

Part C:

```bash
PYTHONPATH=partC python partC/verify_ray.py | tee partC/verify_output.txt
```

Part D (with the Part A server running):

```bash
python partD/smoke_eta.py --url http://localhost:8000 --expect-version v1
```

It exits with 0 if all checks pass and 1 otherwise.
