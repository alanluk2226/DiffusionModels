# Initial Report（結構示意｜精簡版）

**Hong Kong Metropolitan University**  
School of Science and Technology

| | |
|---|---|
| **Name** | [Your Name] |
| **Student ID** | [HKMU ID] |
| **Programme** | COMP S456F / 4560SEF / 4570SEF |
| **Supervisor** | [Supervisor Name] |
| **Date** | September 2026 |
| **Project Title** | From-Scratch Learning of VAEs and Denoising Diffusion Models in PyTorch |

---

## Table of Contents

1. Chapter 1. Introduction (Problem Definition)  
 1.1 Introduction  
 1.2 Project Aim  
 1.3 Project Objectives  
 1.4 Value Propositions  
2. Chapter 2. Background or Literature Review  
 2.1 Review of Existing or Related Solutions  
 2.2 Highlight of the Proposed Solution  
3. Chapter 3. Preliminary Methodology  
 3.1 Overview  
 3.2 Requirements, Supporting Technologies, and Technical Gap  
 3.3 Architecture or High-Level System Design  
4. References  
5. Appendix A. Project Plan  

---

# Chapter 1. Introduction (Problem Definition)

## 1.1 Introduction

Generative models learn a data distribution and can sample new images. Two foundations are the **Variational Autoencoder (VAE)** and **Denoising Diffusion Probabilistic Models (DDPM)**. Many tutorials hide the maths behind high-level APIs. This project studies the equations first, then implements both models **from scratch in PyTorch**.

## 1.2 Project Aim

**Aim:** To understand VAE and diffusion models at the equation level, and to implement both from scratch so that training and sampling behaviour can be verified on simple image datasets (e.g. MNIST / Fashion-MNIST).

## 1.3 Project Objectives

1. **Study** ELBO, VAE, and the link from hierarchical VAE to diffusion.  
2. **Study** DDPM forward/reverse processes and the simplified noise-prediction loss.  
3. **Design and implement** a VAE from scratch (encoder, reparameterization, decoder, KL).  
4. **Design and implement** a DDPM from scratch (noise schedule, U-Net, sampling).  
5. **Train and evaluate** both models with reconstruction / sample quality checks.  
6. **Document** equations, design choices, and experimental results in the final report.

## 1.4 Value Propositions

- **Immediate:** Solid maths + working from-scratch code (not only library wrappers).  
- **Ripple:** A reusable baseline for later work (conditional generation, latent diffusion, etc.) and stronger interview / FYP demonstration material.

---

# Chapter 2. Background or Literature Review

> PPT 版完整文案（含方程投影片）：[`docs/FYP_PPT_Literature_Review.md`](./FYP_PPT_Literature_Review.md)

## 2.1 Review of Existing or Related Solutions

### 2.1.1 Generative modelling families

Classic autoencoders learn a bottleneck reconstruction mapping but do not place a proper prior on the latent code, so ancestral sampling is ill-defined. **Variational Autoencoders (VAEs)** [Kingma & Welling, 2014; Rezende et al., 2014] address this by introducing an approximate posterior \(q_\phi(z|x)\) and maximising the Evidence Lower Bound (ELBO):

\[
\mathrm{ELBO}
=
\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big]
-
D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p(z)\big).
\]

With a diagonal-Gaussian encoder and standard normal prior, the KL term has a closed form; the **reparameterisation trick** \(z=\mu+\sigma\odot\varepsilon\) enables end-to-end SGD. Surveys note that pixel-space VAEs remain stable and likelihood-oriented, yet samples are often blurrier than adversarial models [Kingma & Welling, 2019].

**Generative Adversarial Networks (GANs)** [Goodfellow et al., 2014] produce sharp images via a min–max game, but training can be unstable and they do not supply the explicit ELBO story required for this FYP’s learning goals.

### 2.1.2 Diffusion models

**Diffusion / nonequilibrium** generative models were introduced by Sohl-Dickstein et al. [2015]. **Denoising Diffusion Probabilistic Models (DDPM)** [Ho et al., 2020] made the approach practical for high-quality image synthesis. A fixed forward process injects Gaussian noise:

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\]

and a neural network learns the reverse transitions. Matching reverse Gaussians to the true posterior \(q(x_{t-1}|x_t,x_0)\) yields (up to weighting) the simplified noise-prediction loss \(\mathcal{L}_{\mathrm{simple}}=\mathbb{E}\|\varepsilon-\varepsilon_\theta(x_t,t)\|_2^2\). Score-based / SDE views [Song et al., 2021] unify continuous-time diffusion; **Luo [2022]** further shows that diffusion is a deep **Markovian hierarchical VAE** with a *fixed* Gaussian encoder and a *learned* denoiser—hence VAE study is a premise of diffusion understanding. Latent Diffusion Models [Rombach et al., 2022] combine a VAE compressor with diffusion in latent space and underpin systems such as Stable Diffusion; they are influential but too large as a first from-scratch target.

### 2.1.3 Tooling and tutorials

High-level libraries (e.g. Hugging Face Diffusers) and many Kaggle notebooks deliver fast visual demos but typically bury the ELBO / schedule / loss derivation. That creates a **pedagogical gap**: learners can generate images without connecting each loss term to code.

| Approach | Idea | Limitation for this FYP |
|----------|------|-------------------------|
| Classic Autoencoder | Compress → reconstruct | Latent not regularised for sampling |
| VAE [Kingma & Welling, 2014] | ELBO = recon − KL | Blurry samples; maths often skipped in tutorials |
| GAN [Goodfellow et al., 2014] | Adversarial training | Unstable; weaker likelihood story |
| DDPM [Ho et al., 2020] | Learn to reverse a noise chain | Slow sampling; easy to treat as black-box U-Net |
| Score / SDE [Song et al., 2021] | Estimate ∇ log *p* | Extra abstraction beyond core FYP depth |
| LDM [Rombach et al., 2022] | Diffusion in VAE latent space | Production scale; not first-principles friendly |
| High-level libraries | Fast pretrained pipelines | Hide equations this project must master |

**Domain requirements (lessons learned → design criteria):**

| ID | Requirement |
|----|-------------|
| R1 | Tractable training objective (ELBO / \(\mathcal{L}_{\mathrm{simple}}\)) |
| R2 | Explicit sampling path (decode \(z\) / reverse denoising chain) |
| R3 | Transparent maths–code mapping |
| R4 | Runnable on limited student hardware (MNIST-scale) |
| R5 | Cover **both** VAE and DDPM, with VAE as the premise |

## 2.2 Highlight of the Proposed Solution

| | Existing tutorials / libs | SOTA (e.g. LDM) | This project |
|--|---------------------------|-----------------|--------------|
| Maths | Often skipped or partial | Assumed known | Derive and state key equations |
| Code | Heavy frameworks | Large pretrained stacks | Minimal from-scratch PyTorch |
| Scope | Either VAE *or* diffusion | Full latent-diffusion pipeline | Both, VAE as premise of diffusion |
| Goal | Pretty samples first | FID / production quality | Understanding first, then verified implementation |
| Hardware | Often cloud GPU | Multi-GPU | CPU / single-GPU friendly |

Honest gap vs SOTA: sample quality and speed will not match Stable Diffusion; the value is **clarity and correctness of first principles**, judged against R1–R5.

---

# Chapter 3. Preliminary Methodology

## 3.1 Overview

```
Study ELBO/VAE  →  Study DDPM equations  →  Code VAE  →  Code DDPM  →  Evaluate & write up
```

Objectives map to: theory modules → two model modules → evaluation module → documentation.

## 3.2 Requirements, Supporting Technologies, and Technical Gap

**Use cases**
- Student / researcher: train VAE & DDPM, inspect losses, generate sample grids.  
- Examiner: verify that equations in the report match the code.

**Functional requirements**
- F1 Train VAE; show recon + random samples  
- F2 Train DDPM; show denoising trajectory + final samples  
- F3 Log loss curves; save checkpoints  

**Key technologies:** Python, PyTorch, NumPy/Matplotlib (or similar), MNIST-family datasets.

**Technical gap:** Bridging textbook ELBO/DDPM derivations to a small, correct codebase without relying on `diffusers`-style abstractions.

## 3.3 Architecture or High-Level System Design

### Component diagram (示意)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Dataset    │────▶│  Train Loop  │────▶│ Checkpoints │
│  (MNIST…)   │     │  + Losses    │     │ + Samples   │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐        ┌────────────────┐
     │  VAE Module    │        │  DDPM Module   │
     │ Enc / Dec / KL │        │ Schedule/UNet  │
     └────────────────┘        └────────────────┘
```

### Data-flow diagram (示意)

**VAE:** `x → Encoder(μ,σ) → z (reparam) → Decoder → x̂` Loss = recon + KL  

**DDPM:** `x₀ → add noise (t) → x_t → ε_θ(x_t,t) → MSE(ε,ε_θ)` Sample: `x_T → … → x₀`

*(Final report will replace ASCII with proper drawn diagrams.)*

---

# References

1. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.  
2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. *ICML*.  
3. Kingma, D. P., & Welling, M. (2019). An Introduction to Variational Autoencoders. *Foundations and Trends in Machine Learning*, 12(4), 307–392.  
4. Goodfellow, I., et al. (2014). Generative Adversarial Nets. *NeurIPS*.  
5. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. *ICML*.  
6. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.  
7. Song, Y., et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*.  
8. Luo, C. (2022). Understanding Diffusion Models: A Unified Perspective. *arXiv:2208.11970*.  
9. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*.

---

# Appendix A. Project Plan

| Week | Task | Deliverable |
|------|------|-------------|
| 1–2 | VAE maths + paper notes | Equation sheet |
| 3–4 | VAE from scratch | Trainable VAE + sample grid |
| 5–7 | DDPM maths + Luo notes | Equation sheet |
| 8–10 | DDPM from scratch | Trainable DDPM + samples |
| 11–12 | Compare, ablate, write Initial/Final reports | Report drafts |

*(Group projects: add Appendix B Roles & C Meeting Minutes.)*

---

**— End of mock Initial Report (structure only) —**
