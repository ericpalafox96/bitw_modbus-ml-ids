"""
File: ai_model/train_autoencoder.py
Purpose: Train a simple baseline Autoencoder on packet features to learn
         "normal" behavior. High reconstruction error => anomaly.

Run:
    python ai_model/train_autoencoder.py

Outputs:
    ai_model/model.pt  (weights + normalization stats)

Tip:
- If you don't have pcaps yet, it will auto-generate synthetic data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Local import (same folder)
from features import pcap_to_matrix, FEATURES


class Autoencoder(nn.Module):
    """Tiny MLP autoencoder for tabular features."""
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


def standardize(X: np.ndarray):
    """Return standardized X plus (mu, sigma) for later inference."""
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sig, (mu.astype("float32"), sig.astype("float32"))


def load_training_matrix() -> np.ndarray:
    """Try to load a pcap, else generate synthetic 'normal' data."""
    pcap = Path("data/raw/sample.pcap")
    if pcap.exists():
        try:
            X = pcap_to_matrix(str(pcap), limit=5000)
            if len(X):
                return X
        except Exception:
            pass
    # Fallback synthetic normal-ish data
    rng = np.random.default_rng(42)
    lengths = rng.normal(400, 50, 5000)      # bytes
    protos  = rng.choice([6, 17], 5000)      # TCP/UDP-ish
    sports  = rng.integers(1000, 65000, 5000)
    dports  = rng.integers(1, 1024, 5000)
    iats    = np.abs(rng.normal(3, 1, 5000)) # ms
    X = np.vstack([lengths, protos, sports, dports, iats]).T.astype("float32")
    return X


if __name__ == "__main__":
    X = load_training_matrix().astype("float32")
    Xs, (mu, sig) = standardize(X)

    model = Autoencoder(n_features=Xs.shape[1])
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    Xt = torch.tensor(Xs)
    for epoch in range(20):
        opt.zero_grad()
        out = model(Xt)
        loss = crit(out, Xt)
        loss.backward()
        opt.step()
        print(f"epoch {epoch+1:02d}  loss={loss.item():.6f}")

    save_path = Path("ai_model/model.pt")
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sig": sig, "features": FEATURES},
               save_path)
    print(f"✅ saved {save_path}")
