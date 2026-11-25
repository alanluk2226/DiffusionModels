"""Visualizations for 2D VAE on MNIST (latent map, 4→0 walk, recon blur)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import VAE2D


DIGIT_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}


def load_model(ckpt: Path, device: torch.device) -> VAE2D:
    blob = torch.load(ckpt, map_location=device, weights_only=True)
    model = VAE2D(latent_dim=blob.get("latent_dim", 2), hidden=blob.get("hidden", 400))
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def collect_latents(
    model: VAE2D,
    loader: DataLoader,
    device: torch.device,
    max_points: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    zs, ys = [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        mu = model.encode_mu(x).cpu().numpy()
        zs.append(mu)
        ys.append(y.numpy())
        n += x.size(0)
        if n >= max_points:
            break
    z = np.concatenate(zs, axis=0)[:max_points]
    y = np.concatenate(ys, axis=0)[:max_points]
    return z, y


@torch.no_grad()
def mean_latent_for_digit(
    model: VAE2D,
    dataset: datasets.MNIST,
    digit: int,
    device: torch.device,
    n: int = 200,
) -> torch.Tensor:
    xs = []
    for img, label in dataset:
        if label == digit:
            xs.append(img)
            if len(xs) >= n:
                break
    batch = torch.stack(xs, dim=0).to(device)
    return model.encode_mu(batch).mean(dim=0)


def plot_latent_scatter(z: np.ndarray, y: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for d in range(10):
        m = y == d
        ax.scatter(
            z[m, 0],
            z[m, 1],
            s=8,
            alpha=0.45,
            c=DIGIT_COLORS[d],
            label=str(d),
            rasterized=True,
        )
    ax.set_title("MNIST 2D latent space (encoder μ)")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.legend(title="digit", markerscale=2, loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_interpolation(
    model: VAE2D,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    out: Path,
    steps: int = 12,
    start_label: int = 4,
    end_label: int = 0,
) -> None:
    device = next(model.parameters()).device
    alphas = torch.linspace(0, 1, steps, device=device)
    zs = torch.stack([(1 - a) * z_start + a * z_end for a in alphas], dim=0)
    imgs = model.decode(zs).cpu().numpy()

    fig, axes = plt.subplots(1, steps, figsize=(1.2 * steps, 1.6))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(str(start_label), fontsize=10)
        elif i == steps - 1:
            ax.set_title(str(end_label), fontsize=10)
        else:
            ax.set_title("→", fontsize=9)
    fig.suptitle(
        f"Linear interpolation in latent space: {start_label} → {end_label}",
        y=1.05,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(
    model: VAE2D,
    dataset: datasets.MNIST,
    out: Path,
    n_per_digit: int = 1,
) -> None:
    device = next(model.parameters()).device
    # one example per digit 0–9
    chosen = {}
    for img, label in dataset:
        if label not in chosen:
            chosen[label] = img
        if len(chosen) == 10:
            break
    labels = list(range(10))
    xs = torch.stack([chosen[d] for d in labels], dim=0).to(device)
    with torch.no_grad():
        recons, _, _ = model(xs)
    xs_np = xs.cpu().numpy()
    rc_np = recons.cpu().numpy()

    fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
    for i, d in enumerate(labels):
        axes[0, i].imshow(xs_np[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(str(d), fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(rc_np[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("recon\n(often blurry)", fontsize=8)
    fig.suptitle("VAE reconstruction quality on MNIST", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_overlap_focus(z: np.ndarray, y: np.ndarray, out: Path, digits=(4, 9, 0)) -> None:
    """Highlight morphologically similar digits (e.g. 4 vs 9) and walk endpoint 0."""
    fig, ax = plt.subplots(figsize=(7, 6))
    # background: all digits faint
    ax.scatter(z[:, 0], z[:, 1], s=4, c="#dddddd", alpha=0.35, rasterized=True)
    for d in digits:
        m = y == d
        ax.scatter(
            z[m, 0],
            z[m, 1],
            s=12,
            alpha=0.55,
            c=DIGIT_COLORS[d],
            label=f"digit {d}",
            rasterized=True,
        )
    # mean centers
    for d in digits:
        m = y == d
        cx, cy = z[m, 0].mean(), z[m, 1].mean()
        ax.scatter([cx], [cy], s=120, marker="*", c=DIGIT_COLORS[d], edgecolors="k", zorder=5)
        ax.annotate(str(d), (cx, cy), textcoords="offset points", xytext=(6, 6), fontsize=12)

    # draw mean-to-mean line 4 → 0 (matches whiteboard demo)
    if 4 in digits and 0 in digits:
        c4 = z[y == 4].mean(axis=0)
        c0 = z[y == 0].mean(axis=0)
        ax.plot([c4[0], c0[0]], [c4[1], c0[1]], "k--", lw=1.5, label="4 → 0 mean path")

    ax.set_title("Latent neighbourhood: similar shapes (4/9) & 4→0 path")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("outputs/vae_2d_mnist/vae2d_mnist.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/vae_2d_mnist"))
    p.add_argument("--max-points", type=int, default=8000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.ToTensor()
    test_ds = datasets.MNIST(str(args.data_dir), train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = load_model(args.ckpt, device)
    z, y = collect_latents(model, test_loader, device, max_points=args.max_points)

    plot_latent_scatter(z, y, args.out_dir / "latent_scatter.png")
    plot_overlap_focus(z, y, args.out_dir / "latent_4_9_0_focus.png", digits=(4, 9, 0))

    z4 = mean_latent_for_digit(model, test_ds, 4, device)
    z0 = mean_latent_for_digit(model, test_ds, 0, device)
    plot_interpolation(model, z4, z0, args.out_dir / "interp_4_to_0.png", steps=12)

    plot_reconstructions(model, test_ds, args.out_dir / "reconstructions.png")

    print(f"figures written to {args.out_dir.resolve()}")
    for name in (
        "latent_scatter.png",
        "latent_4_9_0_focus.png",
        "interp_4_to_0.png",
        "reconstructions.png",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
