"""
File: runtime/bridge_mock.py
Purpose: Emulate the BITW device by reading packets (from a pcap or
         synthetic generator), converting to features, and calling the
         inference API. Useful for end-to-end testing without hardware.

Run:
    1) Start the API:
       uvicorn runtime.inference_service:app --reload --port 8000
    2) Stream features:
       python runtime/bridge_mock.py
"""

import time
import json
import requests
import numpy as np
from pathlib import Path
from ai_model.features import pcap_to_matrix, FEATURES


def get_matrix() -> np.ndarray:
    """Load from pcap or synthesize if none available."""
    pcap = Path("data/raw/sample.pcap")
    if pcap.exists():
        try:
            X = pcap_to_matrix(str(pcap), limit=1000)
            if len(X):
                return X
        except Exception:
            pass
    # synth fallback
    rng = np.random.default_rng(7)
    return np.column_stack([
        rng.normal(420, 40, 1000),    # pkt_len
        rng.choice([6, 17], 1000),    # ip_proto
        rng.integers(1000, 65000, 1000),
        rng.integers(1, 1024, 1000),
        np.abs(rng.normal(3, 1, 1000)),
    ]).astype("float32")


if __name__ == "__main__":
    X = get_matrix()
    url = "http://127.0.0.1:8000/infer"
    print(f"Streaming {len(X)} records → {url}")
    for row in X:
        payload = {"values": [float(v) for v in row.tolist()]}
        r = requests.post(url, json=payload, timeout=2)
        res = r.json()
        tag = "ALERT" if res["anomaly"] else "OK"
        print(f"{tag}  err={res['recon_error']:.4f}  {json.dumps(payload)}")
        time.sleep(0.005)  # mimic real-time
