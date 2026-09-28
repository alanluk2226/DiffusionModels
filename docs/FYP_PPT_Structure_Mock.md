# FYP Presentation Mock
## VAE & Diffusion Models from Scratch (PyTorch)

> PPT 必含四塊：(1) Problem statement (2) Aims & objectives (3) Literature review (4) Preliminary methodology  
> 以下為簡短投影片文案示意（約 8–10 頁）

---

# Slide 1 — Title

**From-Scratch Learning of VAEs and Denoising Diffusion Models in PyTorch**

- Name: [Your Name]  
- Student ID: [HKMU ID]  
- Programme: COMP S456F / 4560SEF / 4570SEF  
- Supervisor: [Name]  
- Date: Sep 2026  

---

# Slide 2 — (1) Problem Statement

**Problem**

Generative models (VAE, diffusion) are widely used, but many learners only call high-level libraries and never connect **equations ↔ training code**.

**Why it matters**

Without that link, it is hard to debug training, justify design choices, or extend models later.

**This project addresses**

Understand VAE & DDPM maths first, then implement **both from scratch in PyTorch** on a simple image dataset, so every loss term maps to code.

---

# Slide 3 — (2) Project Aim

**Aim**

To master the core equations of Variational Autoencoders and Denoising Diffusion Probabilistic Models, and to implement both models from scratch in PyTorch with verifiable training and sampling results.

---

# Slide 4 — (2) Project Objectives

1. Study ELBO / VAE and how hierarchical VAEs lead to diffusion  
2. Study DDPM forward–reverse process and the noise-prediction loss  
3. Design & implement a **VAE from scratch**  
4. Design & implement a **DDPM from scratch**  
5. Train, evaluate (loss curves, reconstructions / samples), and document results  

*(5 measurable objectives — fits FYP guidance)*

---

# Slide 5 — (3) Literature Review · Landscape

| Family | Idea | Issue for learning |
|--------|------|--------------------|
| Autoencoder | Compress → reconstruct | Cannot sample a proper prior |
| **VAE** | ELBO = recon − KL | Maths often skipped; blurry samples |
| GAN | Adversarial generator | Unstable; different story |
| **DDPM** | Learn reverse of noise chain | Heavy; easy to treat as black box |
| Libraries (e.g. Diffusers) | Fast demos | Hide equations this FYP must show |

---

# Slide 6 — (3) Literature Review · Key takeaways

- **VAE** = learn encoder \(q_\phi(z|x)\) + decoder \(p_\theta(x|z)\); train with ELBO  
- **Diffusion** ≈ deep Markovian VAE with **fixed Gaussian noising** and a **learned denoiser**  
- Practical DDPM objective: predict noise \(\varepsilon\) added to \(x_0\)  
- Requirement for this FYP: transparent maths–code mapping, not SOTA image quality  

---

# Slide 7 — (3) vs Proposed Solution

| | Typical tutorial / library | This project |
|--|----------------------------|--------------|
| Focus | Pretty samples | Equations + correct minimal code |
| Scope | VAE *or* diffusion | **Both** (VAE as premise) |
| Stack | High-level APIs | From-scratch PyTorch |
| Success | FID / demos | Working models + clear write-up |

---

# Slide 8 — (4) Preliminary Methodology · Overview

```
Study VAE  →  Study DDPM  →  Code VAE  →  Code DDPM  →  Evaluate & report
```

- **Dataset:** MNIST / Fashion-MNIST (small, CPU/GPU friendly)  
- **Stack:** Python, PyTorch, Matplotlib  
- **Eval:** loss curves, reconstructions, sample grids, denoising trajectory  

---

# Slide 9 — (4) Preliminary Methodology · Design

**VAE module**  
`x → Encoder(μ, σ) → z (reparam) → Decoder → x̂`  
Loss = reconstruction + KL

**DDPM module**  
`x₀ → x_t (noise schedule) → ε_θ(x_t, t)`  
Loss = MSE(ε, ε_θ) · Sample: `x_T → … → x₀`

**Architecture (one diagram on the real PPT)**  
Dataset → Train loop → {VAE | DDPM} → Checkpoints + sample images

---

# Slide 10 — Summary & Next Steps

| Done / planned | Item |
|----------------|------|
| Study | VAE + Luo diffusion notes / talks |
| Build | From-scratch VAE, then DDPM |
| Deliver | Working demos + Initial / Final reports |

**Q & A**

---

## Speaker notes（給你對稿用，不必上投影片）

- Problem：不是「做一個生成 App」，而是「方程能講清、程式能對上」。  
- Aim 一句；Objectives 用动词（Study / Design / Implement / Evaluate）。  
- Lit review：表格式最省時間；強調 VAE → diffusion 的連結。  
- Methodology：一條 pipeline + 兩個模組公式就夠；正式版再補 component / data-flow 圖。
