# ETA service. Run from project root:
# ETA_MODEL=data/eta_v1.joblib uvicorn --app-dir partA eta_app:app --port 8000
import os
import time
import uuid
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, model_validator

bundle = joblib.load(os.getenv("ETA_MODEL", "data/eta_v1.joblib"))
model, FEATURES, VERSION = bundle["model"], bundle["feature_columns"], bundle["version"]

registry = CollectorRegistry()
REQUESTS = Counter("eta_requests", "requests", ["path", "status"], registry=registry)
LATENCY = Histogram("eta_request_latency_seconds", "latency", registry=registry,
                    buckets=(.001, .0025, .005, .01, .025, .05, .08, .1, .25, .5, 1))


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def check_feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible: distance_km > 150 and n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)


app = FastAPI()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        resp = JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    resp.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        REQUESTS.labels(path=request.url.path, status=str(resp.status_code)).inc()
        LATENCY.observe(time.perf_counter() - t0)
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(x) for x in e["loc"] if x != "body"), "msg": e["msg"]}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details,
                                                  "request_id": request.state.request_id})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = request.state.request_id
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
        preds = [round(float(p), 1) for p in model.predict(X)]
    except Exception:
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    return {"predictions": preds, "model_version": VERSION, "request_id": rid}
