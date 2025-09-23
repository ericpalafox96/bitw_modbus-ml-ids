# train_simple_autoencoder.py
# ----------------------------------------------------------
# This script:
#  1. Creates synthetic training data
#  2. Defines a very simple Autoencoder neural network
#  3. Trains the model for a few epochs
#  4. Saves the trained model (PyTorch format)
#  5. Exports the model to ONNX format (for Jetson deployment)
# ----------------------------------------------------------

import os, time
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------
# Step 1: Make synthetic data
# ----------------------------------------------------------
def make_synth(n=5000, dim=10):
    """
    Generates a dataset of random numbers shaped like [n, dim].
    - Think of 'n' as the number of samples (rows)
    - 'dim' is the number of features (columns)
    We use Gaussian random numbers just to test training works.
    """
    x = np.random.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
    return torch.from_numpy(x)

# ----------------------------------------------------------
# Step 2: Define the model (Autoencoder)
# ----------------------------------------------------------
class TinyAE(nn.Module):
    """
    Autoencoder = Encoder + Decoder
    - Encoder compresses input into a smaller representation (latent space)
    - Decoder reconstructs the input from that compressed version
    Goal: learn to reproduce the input. If it can’t, that means input is "anomalous".
    """
    def __init__(self, dim):
        super().__init__()
        # Encoder reduces input dimension down to 4
        self.encoder = nn.Sequential(
            nn.Linear(dim, 16),  # shrink from dim -> 16
            nn.ReLU(),
            nn.Linear(16, 4)     # shrink from 16 -> 4 (latent size)
        )
        # Decoder rebuilds back to the original dim
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),    # expand from 4 -> 16
            nn.ReLU(),
            nn.Linear(16, dim)   # expand from 16 -> dim
        )

    def forward(self, x):
        # Data flows through encoder, then decoder
        return self.decoder(self.encoder(x))

# ----------------------------------------------------------
# Step 3: Training loop
# ----------------------------------------------------------
def train(model, data, epochs=10, batch=64, lr=1e-3):
    """
    Trains the model on the provided data.
    - 'epochs' = how many times we loop over the entire dataset
    - 'batch' = how many samples we process at once
    - 'lr' = learning rate (how big each update step is)
    """
    # Optimizer updates model weights based on gradients
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Loss function = mean squared error (MSE) between input and reconstruction
    loss_fn = nn.MSELoss()

    n = data.shape[0]  # total number of samples
    for ep in range(1, epochs+1):
        # Shuffle dataset each epoch
        idx = torch.randperm(n)
        total_loss = 0.0

        # Process data in small batches
        for i in range(0, n, batch):
            b = data[idx[i:i+batch]]

            # Forward pass: run batch through model
            recon = model(b)
            # Compute reconstruction error
            loss = loss_fn(recon, b)

            # Backward pass: compute gradients
            opt.zero_grad()
            loss.backward()
            # Update weights
            opt.step()

            total_loss += loss.item() * b.size(0)

        # Print average loss per epoch (progress indicator)
        avg_loss = total_loss / n
        print(f"Epoch {ep:02d}/{epochs} - loss: {avg_loss:.6f}")

# ----------------------------------------------------------
# Step 4: Export helpers
# ----------------------------------------------------------
def export(model, dim, out_onnx):
    """
    Exports the trained model to ONNX format.
    - ONNX = portable model format you can run on Jetson with TensorRT
    """
    model.eval()  # put model in "inference" mode
    dummy = torch.randn(1, dim)  # fake input with correct shape
    torch.onnx.export(
        model, dummy, out_onnx,
        input_names=["input"], output_names=["recon"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch"}, "recon": {0: "batch"}}
    )

# ----------------------------------------------------------
# Step 5: Main script entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("artifacts", exist_ok=True)

    dim = 10  # number of input features (columns)
    # Make dataset
    data = make_synth(5000, dim)

    # Create model
    model = TinyAE(dim)

    # Train the model
    train(model, data, epochs=10)

    # Save PyTorch weights
    torch.save(model.state_dict(), "artifacts/tiny_autoencoder.pt")

    # Export to ONNX
    export(model, dim, "artifacts/tiny_autoencoder.onnx")

    print("Saved artifacts to ./artifacts (pt + onnx). Took %.1fs" % (time.time()-t0))
