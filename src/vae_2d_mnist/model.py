"""2D latent VAE for MNIST (from scratch)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden: int = 400) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden: int = 400) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class VAE2D(nn.Module):
    """Encoder(μ,σ) → reparameterize → Decoder → x̂."""

    def __init__(self, latent_dim: int = 2, hidden: int = 400) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim, hidden=hidden)
        self.decoder = Decoder(latent_dim=latent_dim, hidden=hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = μ + σ ⊙ ε, ε ~ N(0, I)  — makes sampling differentiable
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO terms: reconstruction (BCE) + β · KL(q(z|x) || N(0,I))."""
    bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    # KL = -½ Σ (1 + log σ² − μ² − σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return bce + beta * kl, bce, kl
