"""
File: runtime/inference_service.py
Purpose: Provide a simple HTTP API for anomaly scoring using the trained
         autoencoder model. The BITW (or a mock bridge) will call this.

Run (dev):
    uvicorn runtime.inference_service:app --reload --port 8000

Test:
    curl -X POST http://127.0.0.1:8000/infer -H "Content-Type: application/json" \
         -d '{"values":[450,6,443,502,2.5]}'
"""

from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class Autoencoder(nn.Module):
    """Must mirror architecture from training script."""
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, n_features)
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


class FeatureVec(BaseModel):
    """One observation: [pkt_len, ip_proto, src_port, dst_port, iat_ms]."""
    values: List[float]


# Load model & normalization stats
PACK = torch.load(Path("ai_model/model.pt"), map_location="cpu")
MU, SIG, FEATS = PACK["mu"], PACK["sig"], PACK.get("features", [])
MODEL = Autoencoder(n_features=int(MU.shape[1]))
MODEL.load_state_dict(PACK["state_dict"])
MODEL.eval()

# Default threshold—tune later using validation data
THRESH = 3.0

app = FastAPI(title="Edge AI Inference", version="0.1.0")


def score(vec: list[float]) -> tuple[float, bool]:
    """Return (reconstruction_error, is_anomaly)."""
    x = (np.array(vec, dtype=np.float32) - MU) / SIG
    xin = torch.tensor(x[None, :])
    with torch.no_grad():
        rec = MODEL(xin).numpy()[0]
    err = float(((rec - x) ** 2).mean())
    return err, (err > THRESH)


@app.get("/")
def root():
    return {"status": "ok", "features": FEATS or "unknown", "threshold": THRESH}


@app.post("/infer")
def infer(f: FeatureVec):
    err, anom = score(f.values)
    return {"recon_error": err, "anomaly": bool(anom), "threshold": THRESH}
