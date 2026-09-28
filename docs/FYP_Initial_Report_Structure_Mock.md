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

## 2.1 Review of Existing or Related Solutions

| Approach | Idea | Limitation for this FYP |
|----------|------|-------------------------|
| Classic Autoencoder | Compress → reconstruct | Latent space not regularised for sampling |
| VAE [Kingma & Welling] | ELBO = recon − KL | Blurry samples; needs careful β / architecture |
| GAN | Adversarial training | Unstable; weaker likelihood story |
| DDPM [Ho et al.] | Learn to reverse a noise chain | Slow sampling; heavier compute |
| High-level libraries (diffusers, etc.) | Fast results | Hide the equations this project must master |

**Domain requirements (from lessons learned):**  
(R1) Tractable training objective (R2) Explicit sampling path (R3) Transparent maths–code mapping (R4) Runnable on limited hardware.

## 2.2 Highlight of the Proposed Solution

| | Existing tutorials / libs | This project |
|--|---------------------------|--------------|
| Maths | Often skipped or partial | Derive and state key equations |
| Code | Heavy frameworks | Minimal from-scratch PyTorch |
| Scope | Either VAE *or* diffusion | Both, with VAE treated as the premise of diffusion |
| Goal | Pretty samples first | Understanding first, then verified implementation |

Honest gap vs SOTA: sample quality and speed will not match Stable Diffusion; the value is **clarity and correctness of first principles**.

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
2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.  
3. Luo, C. (2022). Understanding Diffusion Models: A Unified Perspective. *arXiv:2208.11970*.  
4. [Add ≥3 more journal/conference papers for the real submission]

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
