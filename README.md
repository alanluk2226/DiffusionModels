# DiffusionModels

HKMU Final Year Project workspace: study and from-scratch implementation of **VAE** and **Denoising Diffusion (DDPM)** in PyTorch.

## Goal

1. Master the core equations of VAE and diffusion models.
2. Implement both **from scratch** in this repo (not only high-level library demos).
3. Document understanding in reports / presentations.

## Demo: 2D VAE on MNIST (latent space)

Train a VAE with **latent_dim = 2**, then observe:

| Figure | What to look at |
|--------|------------------|
| `artifacts/vae_2d_mnist/latent_scatter.png` | Digit clusters in 2D latent plane |
| `artifacts/vae_2d_mnist/latent_4_9_0_focus.png` | Similar shapes **4 / 9** nearby; **4 → 0** mean path |
| `artifacts/vae_2d_mnist/interp_4_to_0.png` | Linear interpolation: smooth **4 → 0** morph |
| `artifacts/vae_2d_mnist/reconstructions.png` | Reconstructions are typically **blurry** |

```bash
pip install -r requirements.txt
python -m src.vae_2d_mnist.run_demo --epochs 15

# visualize only (needs existing checkpoint)
python -m src.vae_2d_mnist.run_demo --skip-train
```

**Google Colab (recommended for quick demo):** open  
[`notebooks/VAE_2D_MNIST_Latent_Demo.ipynb`](notebooks/VAE_2D_MNIST_Latent_Demo.ipynb)  
→ upload to Colab, or use *File → Open notebook → GitHub*, then **Runtime → Run all**.

## Learning path (supervisor brief)

| Step | Material |
|------|----------|
| PyTorch refresh (optional) | [Aladdin Persson playlist 1–18](https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4) |
| VAE (videos 1–3) | [VAE playlist](https://www.youtube.com/playlist?list=PLivJwLo9VCUK1dXFU9Ig96fjANOMdoL9R) |
| VAE intuition | [Why VAE matters](https://www.youtube.com/watch?v=qJeaCHQ1k2w&t=830s) |
| Diffusion paper | [Luo, arXiv:2208.11970](https://arxiv.org/pdf/2208.11970) |
| Diffusion talk | [Same author talk](https://www.youtube.com/watch?v=bGh_uBbj_Po) |
| DDPM code reference | [Kaggle DDPM from scratch](https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch) · [Video](https://www.youtube.com/watch?v=vu6eKteJWew) |
| VAE code | Self-sourced / from scratch |

## Repo layout

```text
docs/                    # PPT / report notes
src/vae_2d_mnist/        # 2D VAE train + visualize
artifacts/vae_2d_mnist/  # Demo figures (committed)
data/                    # MNIST download (gitignored)
outputs/                 # Checkpoints (gitignored)
```

## Status

- Literature review drafted under `docs/`
- **2D VAE MNIST latent-space demo** implemented

## License

MIT (course project materials).
