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

> **Ownership:** Literature Review is the section this contributor owns.  
> PPT 完整文案：[`docs/FYP_PPT_Literature_Review.md`](./FYP_PPT_Literature_Review.md)  
> Supervisor framing：[`docs/supervisor-guidance.md`](./supervisor-guidance.md)

## 2.1 Review of Existing or Related Solutions

### 2.1.1 Generative AI and Large Vision Models (LVM)

Generative AI spans large language models (LLMs) and large vision models (LVMs). A generative LVM typically maps **noise plus optional guidance** (text, class, mask, …) to an **image**. Recent systems have largely **converged on diffusion** as the core generative algorithm [Ho et al., 2020; Song et al., 2021], often combined with multi-modal conditioning. Latent Diffusion Models [Rombach et al., 2022] further show that practical LVMs compose a **VAE perceptual compressor** with **diffusion in latent space**—so VAE literacy is not optional for understanding modern generative vision.

### 2.1.2 VAE foundations

Classic autoencoders lack a proper latent prior for sampling. **VAEs** [Kingma & Welling, 2014; Rezende et al., 2014] introduce \(q_\phi(z|x)\) and maximise the ELBO:

\[
\mathrm{ELBO}
=
\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big]
-
D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p(z)\big).
\]

The reparameterisation trick enables SGD through stochastic latents. Surveys note stability and a clear likelihood story, but often blurrier samples than GANs [Kingma & Welling, 2019]. **GANs** [Goodfellow et al., 2014] produce sharp images yet offer a weaker ELBO narrative for this project’s learning goals.

### 2.1.3 Diffusion models (high-level procedure)

Diffusion generative models originate in nonequilibrium thermodynamics [Sohl-Dickstein et al., 2015]. **DDPM** [Ho et al., 2020] made them practical:

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\quad
\mathcal{L}_{\mathrm{simple}}=\mathbb{E}\|\varepsilon-\varepsilon_\theta(x_t,t)\|_2^2.
\]

**High-level I/O:** train on noisified images; sample from \(x_T\sim\mathcal{N}(0,I)\) by iterative denoising (plus guidance for LVMs). **Luo [2022]** frames diffusion as a deep **Markovian hierarchical VAE** with a *fixed* Gaussian forward process—hence VAE study is the premise of diffusion understanding.

### 2.1.4 Applications: image generation and inpainting

| Task | Representative approaches | Lesson |
|------|---------------------------|--------|
| **Image generation** | VAE/GAN sampling; DDPM ancestral sampling; latent diffusion with text guidance [Rombach et al., 2022] | Same denoising backbone scales from toy data to LVMs |
| **Image inpainting** | GAN/CNN fill-in; **RePaint** mask-constrained diffusion [Lugmayr et al., 2022]; latent inpainting | Reverse diffusion naturally supports constraining known pixels |

Supervisor-aligned scope therefore reviews **both** tasks under one generative-LVM umbrella.

### 2.1.5 Theory vs application settings; tooling gap

Literature commonly uses **MNIST-scale** data to validate algorithms, and **CIFAR / ImageNet** (often with pretrained weights) for real-image applications. High-level libraries (e.g. Diffusers) speed demos but bury ELBO / schedule derivations—creating a pedagogical gap this review targets.

| Approach | Idea | Limitation for this FYP |
|----------|------|-------------------------|
| VAE [Kingma & Welling, 2014] | ELBO = recon − KL | Maths often skipped; blurry pixels |
| GAN [Goodfellow et al., 2014] | Adversarial training | Unstable; weak likelihood story |
| DDPM [Ho et al., 2020] | Reverse noise chain | Easy to treat as black-box U-Net |
| LDM [Rombach et al., 2022] | VAE latent + diffusion | Opaque if only calling APIs |
| RePaint [Lugmayr et al., 2022] | Diffusion inpainting | Extra sampling cost |
| High-level libraries | Fast pretrained pipelines | Hide equations |

**Domain requirements (R1–R6):** (R1) explain LVM I/O (noise+guidance→image); (R2) master VAE ELBO as premise; (R3) state diffusion procedure + key equations; (R4) cover generation **and** inpainting; (R5) separate toy/theory vs pretrained/application tracks; (R6) transparent maths↔PyTorch path.

## 2.2 Highlight of the Proposed Solution

| | Library demo | SOTA LVM | This project’s theory direction |
|--|--------------|----------|--------------------------------|
| Goal | Fast pictures | Beat benchmarks | Understand generative LVM equations |
| VAE | Hidden | Assumed component | Studied explicitly |
| Diffusion | Black-box API | Full system | I/O + procedure + key losses |
| Tasks | Often generation only | Task-specific | **Generation + inpainting** |
| Data | Pretrained only | Web-scale | MNIST theory → pretrained real apps |

Honest gap: from-scratch quality will not match Stable Diffusion. Value: a clear **VAE → diffusion → two vision applications** narrative aligned with how generative LVMs work (R1–R6).

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
10. Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., & Van Gool, L. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models. *CVPR*.

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
