# DiffusionModels

HKMU Final Year Project workspace: study and from-scratch implementation of **VAE** and **Denoising Diffusion (DDPM)** in PyTorch.

## Goal

1. Master the core equations of VAE and diffusion models.
2. Implement both **from scratch** in this repo (not only high-level library demos).
3. Document understanding in reports / presentations.

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
docs/          # Initial report mock, PPT outline, literature review, study notes
src/           # (later) from-scratch VAE & DDPM
```

| Doc | Role |
|-----|------|
| [`docs/supervisor-guidance.md`](docs/supervisor-guidance.md) | Supervisor PPT framing (motivation / method / outcome) |
| [`docs/FYP_PPT_Literature_Review.md`](docs/FYP_PPT_Literature_Review.md) | **Your deliverable:** Literature review slides |
| [`docs/FYP_PPT_Structure_Mock.md`](docs/FYP_PPT_Structure_Mock.md) | Full PPT skeleton (4 blocks; you own lit review only) |
| [`docs/FYP_Initial_Report_Structure_Mock.md`](docs/FYP_Initial_Report_Structure_Mock.md) | Initial report mock (Ch.2 = lit review) |
| [`docs/study-notes.md`](docs/study-notes.md) | Equation study notes |

## Status

**Literature review** aligned to supervisor guidance (Generative LVM → VAE → diffusion; apps = generation + inpainting). Other PPT sections are out of this contributor’s ownership.

## License

MIT (course project materials).
