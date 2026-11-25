"""One-shot: train 2D VAE on MNIST + write latent / interpolation / recon figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="2D VAE MNIST demo (train + visualize)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--skip-train", action="store_true", help="Only visualize using existing ckpt")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    ckpt = root / "outputs" / "vae_2d_mnist" / "vae2d_mnist.pt"
    data = root / "data"
    out = root / "artifacts" / "vae_2d_mnist"

    py = sys.executable
    if not args.skip_train:
        subprocess.check_call(
            [
                py,
                "-m",
                "src.vae_2d_mnist.train",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--data-dir",
                str(data),
                "--ckpt",
                str(ckpt),
            ],
            cwd=str(root),
        )

    subprocess.check_call(
        [
            py,
            "-m",
            "src.vae_2d_mnist.visualize",
            "--ckpt",
            str(ckpt),
            "--data-dir",
            str(data),
            "--out-dir",
            str(out),
        ],
        cwd=str(root),
    )


if __name__ == "__main__":
    main()
