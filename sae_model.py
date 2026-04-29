"""
Sparse Autoencoder for VLM decoder hidden state decomposition.
Standard architecture following Anthropic/OpenAI SAE design:
  z = ReLU(W_enc @ (x - b_dec) + b_enc)
  x_hat = W_dec @ z + b_dec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_enc = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.W_dec = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        z = F.relu(x_centered @ self.W_enc.T + self.b_enc)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-normalize decoder columns (feature directions)."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def compute_metrics(self, x: torch.Tensor) -> dict:
        x_hat, z = self.forward(x)
        mse = F.mse_loss(x_hat, x).item()
        cosine = F.cosine_similarity(x_hat, x, dim=-1).mean().item()
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        l1 = z.abs().sum(dim=-1).mean().item()
        frac_dead = (z.sum(dim=0) == 0).float().mean().item()
        return {
            "mse": mse,
            "cosine_sim": cosine,
            "l0_sparsity": l0,
            "l1_norm": l1,
            "frac_dead_features": frac_dead,
        }
