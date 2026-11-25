"""Train 2D VAE on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .model import VAE2D, vae_loss


def get_loaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(
    model: VAE2D,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> float:
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        recon, mu, logvar = model(x)
        loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loss(model: VAE2D, loader: DataLoader, device: torch.device, beta: float) -> float:
    model.eval()
    total = 0.0
    for x, _ in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def main() -> None:
    p = argparse.ArgumentParser(description="Train 2D VAE on MNIST")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--ckpt", type=Path, default=Path("outputs/vae_2d_mnist/vae2d_mnist.pt"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    model = VAE2D(latent_dim=2, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, args.beta)
        te = eval_loss(model, test_loader, device, args.beta)
        print(f"epoch {epoch:02d}/{args.epochs}  train={tr:.2f}  test={te:.2f}")
        if te < best:
            best = te
            torch.save(
                {
                    "model": model.state_dict(),
                    "latent_dim": 2,
                    "hidden": args.hidden,
                    "epoch": epoch,
                    "test_loss": te,
                },
                args.ckpt,
            )
            print(f"  saved best → {args.ckpt}")

    print("done.")


if __name__ == "__main__":
    main()
