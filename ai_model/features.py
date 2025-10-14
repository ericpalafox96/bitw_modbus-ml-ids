"""
File: ai_model/features.py
Purpose: Convert raw network packets/records into numeric feature vectors
         that an ML model can learn from (and later score in real time).

How to use:
    from ai_model.features import pcap_to_matrix
    X = pcap_to_matrix("data/raw/sample.pcap")

Notes:
- Uses Scapy to read .pcap files when available.
- If you don't have pcaps yet, you can feed synthetic rows to your model.
"""

from typing import Tuple
import numpy as np

try:
    from scapy.all import rdpcap  # optional: only needed if using pcaps
except Exception:
    rdpcap = None  # keeps the module importable without Scapy installed


FEATURES = ["pkt_len", "ip_proto", "src_port", "dst_port", "iat_ms"]


def pkt_to_row(pkt, prev_ts: float | None) -> Tuple[np.ndarray, float]:
    """
    Convert a Scapy packet into a single feature row.

    Args:
        pkt: Scapy packet object.
        prev_ts: timestamp of previous packet (float seconds) or None.

    Returns:
        (row, ts) where row is a 1D float32 array of shape (5,)
        and ts is the current packet timestamp (float).
    """
    ts = float(getattr(pkt, "time", 0.0))
    iat = (ts - prev_ts) * 1000 if prev_ts else 0.0  # inter-arrival time (ms)
    length = len(bytes(pkt)) if hasattr(pkt, "original") or hasattr(pkt, "build") else 0

    # Crude protocol/port extraction—good enough for baseline.
    proto = 6 if hasattr(pkt, "sport") and hasattr(pkt, "dport") else 0
    sport = int(getattr(pkt, "sport", 0))
    dport = int(getattr(pkt, "dport", 0))

    row = np.array([length, proto, sport, dport, iat], dtype=np.float32)
    return row, ts


def pcap_to_matrix(path: str, limit: int | None = None) -> np.ndarray:
    """
    Load a .pcap and convert to an (N x 5) numpy matrix of features.

    Args:
        path: path to pcap file.
        limit: optional max packets to parse.

    Returns:
        np.ndarray of shape (N, 5) with float32 features.
    """
    if rdpcap is None:
        raise RuntimeError("Scapy/rdpcap not available. Install scapy or use synthetic data.")
    pkts = rdpcap(path)
    X, prev = [], None
    for i, p in enumerate(pkts):
        if limit and i >= limit:
            break
        row, prev = pkt_to_row(p, prev)
        X.append(row)
    return np.vstack(X) if X else np.zeros((0, len(FEATURES)), dtype=np.float32)
