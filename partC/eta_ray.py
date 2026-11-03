# Router -> EtaModelV1 (2 replicas) / EtaModelV2 (autoscaling 1-3)
# CPU: 0.1 + 2*0.25 + 3*0.25 = 1.35 <= 2
import hashlib
import os
import time
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve


def bucket(customer_id):
    h = hashlib.sha256(customer_id.encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


@serve.deployment(ray_actor_options={"num_cpus": 0.25}, max_ongoing_requests=16,
                  logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path, extra_work_ms=0.0):
        b = joblib.load(path)
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]
        self.extra = extra_work_ms / 1000

    async def __call__(self, rows):
        p = self.model.predict(pd.DataFrame(rows)[self.cols])
        if self.extra:
            time.sleep(self.extra)
        return {"predictions": [round(float(x), 1) for x in p], "model_version": self.version}


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)


class Req(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: List[Delivery] = Field(min_length=1, max_length=500)


api = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1}, logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class Router:
    def __init__(self, prod, cand, weight):
        self.handles = {"v1": prod, "v2": cand}
        self.weight = weight

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "weight": self.weight}

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: Req, x_beta: Optional[str] = Header(default=None)):
        arm = "v2" if x_beta == "1" or bucket(req.customer_id) < self.weight else "v1"
        out = await self.handles[arm].remote([d.model_dump() for d in req.instances])
        return {**out, "arm": arm}


def build(weight=0.1, extra_work_ms=0.0):
    prod = EtaModel.options(name="EtaModelV1", num_replicas=2).bind(
        os.path.abspath("data/eta_v1.joblib"), extra_work_ms)
    cand = EtaModel.options(name="EtaModelV2", autoscaling_config={
        "min_replicas": 1, "max_replicas": 3, "target_ongoing_requests": 2,
        "upscale_delay_s": 5, "downscale_delay_s": 60}).bind(os.path.abspath("data/eta_v2.joblib"), extra_work_ms)
    return Router.bind(prod, cand, weight)
