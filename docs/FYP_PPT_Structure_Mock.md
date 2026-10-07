# FYP Presentation Mock
## VAE & Diffusion Models from Scratch (PyTorch)

> PPT 必含四塊：(1) Problem statement (2) Aims & objectives (3) Literature review (4) Preliminary methodology  
> **你的負責範圍：僅 (3) Literature review**（對齊 supervisor 指引；見 `supervisor-guidance.md`）。  
> 以下為簡短投影片文案示意；Lit review 以 `FYP_PPT_Literature_Review.md` 為準。

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

# Slides — (3) Literature Review ← **你的部分**

> **完整投影片文案：** [`docs/FYP_PPT_Literature_Review.md`](./FYP_PPT_Literature_Review.md)  
> **Supervisor 指引摘錄：** [`docs/supervisor-guidance.md`](./supervisor-guidance.md)

**建議頁序（對齊 supervisor：LVM / VAE→diffusion / 兩應用）**

| 頁 | 內容 |
|----|------|
| 1 | Scope：Generative AI → LVM → diffusion |
| 2 | LLM vs LVM；noise + guidance → image |
| 3 | Landscape：VAE / GAN / DDPM / LDM |
| 4 | VAE ELBO（premise） |
| 5 | VAE → diffusion bridge（Luo）；high-level I/O |
| 6 | Diffusion procedure + \(\mathcal{L}_{\mathrm{simple}}\) |
| 7 | App：image generation |
| 8 | App：image inpainting（e.g. RePaint） |
| 9 | MNIST theory vs pretrained real apps |
| 10 | Requirements R1–R6 + gap summary |

**一頁版精簡 takeaways**

- LVM：noise (+ guidance) → image；主流算法 = diffusion  
- VAE = 理解 diffusion / latent LVM 的前提  
- 兩應用：generation + inpainting  
- 理論用 toy／from-scratch；真實圖用 pretrained  

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
- Lit review：依 `FYP_PPT_Literature_Review.md`；表格式 + VAE→diffusion 一條線；結尾抽出 R1–R5。  
- Methodology：一條 pipeline + 兩個模組公式就夠；正式版再補 component / data-flow 圖。
