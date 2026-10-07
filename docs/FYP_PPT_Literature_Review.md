# FYP PPT — (3) Literature Review（完整稿）

> **你的負責範圍：** Literature Review only（其餘 Problem / Aims / Methodology / Outcome 依 supervisor 指引由整體 PPT 架構覆蓋）。  
> **對齊 supervisor framing：** Generative AI → LVM（noise + guidance → image）→ diffusion；應用 = **image generation** + **image inpainting**；VAE 為理解 diffusion 的前提。  
> 本檔可直接貼入簡報；對應 Initial Report Chapter 2。引用 ≥6 篇學術文獻。

---

# Slide 1 — Literature Review：Scope & Roadmap

**What this review covers**

| # | Topic | Why it matters for this FYP |
|---|--------|-----------------------------|
| 1 | Generative AI & Large Vision Models (LVM) | Supervisor motivation: LVM path converges to diffusion |
| 2 | VAE foundations | Premise for understanding diffusion maths |
| 3 | Diffusion models (high-level I/O + procedure) | Core of generative LVM |
| 4 | Applications: generation & inpainting | Two target tasks |
| 5 | Gap → domain requirements | Bridge to methodology (not written here) |

**Core claim**

Modern generative LVMs take **noise + (optional) guidance** as input and emit an **image**; the dominant algorithm family is **diffusion**. Understanding diffusion at equation level starts from the **VAE / ELBO** story [Luo, 2022].

---

# Slide 2 — Generative AI Context: LLM vs LVM

| | Large Language Model (LLM) | Large Vision Model (LVM, generative) |
|--|----------------------------|--------------------------------------|
| Typical I/O | Text → text | **Noise (+ guidance)** → **image** |
| Dominant paradigm | Autoregressive / transformer LM | **Diffusion** (also GAN, VAE historically) |
| Conditioning | Prompt tokens | Text / mask / class / image hint |
| Multi-modal trend | LLM + LVM + audio + other modalities in one system | |

**Literature takeaway**

- Generative vision systems increasingly share a **denoising** view: start from noise, iteratively refine toward data [Ho et al., 2020; Song et al., 2021].  
- Latent Diffusion / Stable Diffusion-class models [Rombach et al., 2022] show production LVMs = **VAE compression + diffusion in latent space** + guidance.  
- This FYP’s theory track therefore studies **VAE then diffusion**, matching how real LVMs are built.

---

# Slide 3 — Landscape of Generative Vision Models

| Family | Core idea | Strength | Issue for learning generative LVM |
|--------|-----------|----------|-----------------------------------|
| Autoencoder | Compress → reconstruct | Simple features | No proper sampling prior |
| **VAE** [Kingma & Welling, 2014] | ELBO = recon − KL | Tractable bound; stable | Blurry pixels; maths often skipped |
| GAN [Goodfellow et al., 2014] | Adversarial generator | Sharp images | Unstable; weak likelihood story |
| **DDPM / Diffusion** [Ho et al., 2020] | Reverse a noise chain | Strong samples; clear I/O | Easy to treat as black-box U-Net |
| Score / SDE [Song et al., 2021] | Estimate ∇ log *p* | Continuous unified view | Extra abstraction |
| Latent Diffusion [Rombach et al., 2022] | Diffuse in VAE latent space | Scalable LVM path | Heavy; pretrained-centric |

**Lesson:** To “understand generative LVM in depth,” prioritise **VAE + diffusion** (explicit generative story) over GAN-only or library-only demos.

---

# Slide 4 — VAE: Why It Is the Premise

**Problem:** \(p_\theta(x)=\int p_\theta(x|z)\,p(z)\,dz\) is intractable.

**Solution [Kingma & Welling, 2014; Rezende et al., 2014] — Evidence Lower Bound**

\[
\log p_\theta(x)
=
\mathrm{ELBO}(\theta,\phi;x)
+
D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p_\theta(z|x)\big)
\]

\[
\mathrm{ELBO}
=
\underbrace{\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big]}_{\text{reconstruction}}
-
\underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{prior matching}}
\]

**Reparameterisation:** \(z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon\), \(\varepsilon\sim\mathcal{N}(0,I)\).

**Why VAE matters for LVM / diffusion**

- Teaches **encoder–decoder + regularised latent + sampling from a prior**  
- Production LVMs still use VAEs as the **perceptual compressor** before latent diffusion [Rombach et al., 2022]  
- Hierarchical / Markovian VAEs are the conceptual bridge to diffusion [Luo, 2022]

---

# Slide 5 — From VAE to Diffusion (bridge slide)

**Markovian Hierarchical VAE → Diffusion [Luo, 2022]**

Diffusion can be viewed as a deep Markovian VAE where:

1. Latent dim = data dim (\(z_t \equiv x_t\))  
2. Forward “encoder” = **fixed** Gaussian noising (not learned)  
3. \(x_T \sim \mathcal{N}(0,I)\)  
4. Only the **reverse denoiser** is learned  

**Supervisor emphasis to stress in talk**

- Start from VAE, then generalise to diffusion  
- Clarify the **relationship** (shared ELBO / latent-variable view; diffusion ≈ deep MHVAE with fixed Gaussian forward)  
- For diffusion at lit-review depth: **input / output + high-level procedure** (full algorithm detail later in methodology)

| | VAE (shallow) | Diffusion (DDPM) |
|--|---------------|------------------|
| Input (train) | Image \(x\) | Image \(x_0\), time \(t\), noise \(\varepsilon\) |
| Input (sample) | \(z\sim p(z)\) | \(x_T\sim\mathcal{N}(0,I)\) (+ optional guidance) |
| Output | Reconstruction / sample \(\hat x\) | Denoised sample \(x_0\) |
| Learned part | Encoder + decoder | Denoiser \(\varepsilon_\theta(x_t,t)\) (forward fixed) |

---

# Slide 6 — Diffusion: High-Level Procedure & Key Equations

**High-level procedure (what lit review must own)**

1. **Forward:** gradually add noise to \(x_0\) → \(x_T\) nearly Gaussian  
2. **Train:** learn to predict the noise (or clean image) at random \(t\)  
3. **Sample:** start from noise; iteratively denoise → image  
4. **Guidance (LVM):** text / class / mask steers the reverse process  

**Must-know equations [Ho et al., 2020]**

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon
\]

\[
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\varepsilon}
\big\|\varepsilon-\varepsilon_\theta(x_t,t)\big\|_2^2
\]

**One-sentence mental model**

> Generative LVM ≈ guided reverse diffusion; training maximises an ELBO that becomes “predict the noise.”

---

# Slide 7 — Application 1: Image Generation

| Approach | How generation works | Literature note |
|----------|----------------------|-----------------|
| VAE sampling | \(z\sim p(z)\) → decode | Stable but often blurry [Kingma & Welling, 2019] |
| GAN sampling | Latent → generator | Sharp; training brittle [Goodfellow et al., 2014] |
| **DDPM** | \(x_T\to\cdots\to x_0\) ancestral denoising | Strong unconditional quality [Ho et al., 2020] |
| **Latent Diffusion / LVM** | Diffuse in VAE latent; decode to pixels; text guidance | Scalable generation [Rombach et al., 2022] |

**Lessons for this project**

- Toy theory (e.g. MNIST): from-scratch VAE/DDPM enough to verify equations  
- Real-image LVM generation: literature & practice rely on **pretrained** latent diffusion stacks  
- Conditioning / guidance is what turns a denoiser into a usable generative LVM

---

# Slide 8 — Application 2: Image Inpainting

**Task:** given image \(x\) and mask \(m\), fill missing pixels so the result is realistic and consistent with known regions.

| Approach | Idea | Limitation |
|----------|------|------------|
| Classical / patch methods | Copy similar textures | Fail on semantics / large holes |
| CNN / GAN inpainting | Directly predict missing pixels | Mask-specific training; artefacts |
| **Diffusion inpainting** | Denoise while **keeping known pixels fixed** (or resample them) | Slower; needs careful mask handling |
| **RePaint** [Lugmayr et al., 2022] | Force unmasked pixels during reverse diffusion; improve via resample | Extra compute; sampling schedule sensitive |
| Latent diffusion inpainting | Mask-aware conditioning in latent LVM | Needs pretrained backbone |

**Literature takeaway**

- Diffusion’s iterative reverse process naturally supports **constrained generation** → strong fit for inpainting  
- Same generative LVM backbone can serve **both** generation and inpainting (two applications in supervisor brief)

---

# Slide 9 — Evaluation Settings in Prior Work (ties to intended outcome)

| Setting | Typical use in literature | Role in this FYP (from supervisor) |
|---------|---------------------------|-------------------------------------|
| **MNIST** (toy) | Debug algorithms; verify ELBO / sampling | Train + test to finish **theory learning** |
| CIFAR-10 (32×32) | Standard small natural images | Real-image step; often still trainable |
| ImageNet / 256² | Large-scale LVM benchmark | **Pretrained** models for **applications** |

**Lesson learned:** Papers separate *algorithmic understanding* (small data, from scratch) from *application demonstration* (pretrained LVM on real images). Lit review supports that split; methodology will implement it.

---

# Slide 10 — Lessons → Domain Requirements

| Source | + | − |
|--------|---|---|
| VAE literature | Clear ELBO; reparam; latent sampling | Blurry; shallow latent ≠ full LVM |
| DDPM / score literature | Explicit noise→image path | Slow sampling; maths easy to skip |
| Latent Diffusion / LVM | Practical generation + guidance | Opaque if only calling APIs |
| Inpainting (RePaint et al.) | Mask-constrained reverse process | Schedule / compute overhead |
| Tutorials & Diffusers | Fast demos | Hide equations this review must expose |

**Domain requirements (for judging prior work & guiding later design)**

| ID | Requirement |
|----|-------------|
| R1 | Explain generative **LVM I/O**: noise + guidance → image |
| R2 | Master **VAE ELBO** as premise of diffusion |
| R3 | State diffusion **high-level procedure** + key equations (\(\bar\alpha_t\), \(\mathcal{L}_{\mathrm{simple}}\)) |
| R4 | Cover **two tasks**: image generation & image inpainting |
| R5 | Separate **theory track** (toy/from-scratch) vs **application track** (pretrained real images) |
| R6 | Transparent maths↔code path (PyTorch), not library-only usage |

---

# Slide 11 — Gap vs This Project’s Direction

| | Typical library demo | SOTA LVM paper | **This project (theory focus of lit review)** |
|--|---------------------|----------------|-----------------------------------------------|
| Goal | Pretty pictures fast | Beat FID / user study | Understand generative LVM equations |
| VAE | Hidden preprocessor | Latent encoder assumed | Studied explicitly |
| Diffusion | Black-box pipeline | Full system | I/O + procedure + key losses |
| Tasks | Often generation only | Task-specific | **Generation + inpainting** |
| Data | Pretrained only | Web-scale | MNIST theory → pretrained real apps |

**Honest gap:** this FYP will not match Stable Diffusion quality from scratch.  
**Value claimed in lit review:** a clear path from **VAE → diffusion → two vision applications**, aligned with how generative LVMs actually work.

---

# Slide 12 — Literature Review Summary

1. Generative **LVMs** map **noise (+ guidance) → image**; the field has largely **converged to diffusion**.  
2. **VAE** supplies the ELBO / latent-variable foundation (and remains the compressor inside latent LVMs).  
3. **Diffusion** = deep Markovian VAE with fixed Gaussian forward + learned denoiser; practical loss predicts \(\varepsilon\).  
4. Same backbone supports **image generation** and **image inpainting** (mask-constrained reverse process).  
5. Prior work supports a **two-track** plan: toy/from-scratch for theory; pretrained LVM for real-image applications.  
6. Requirements **R1–R6** hand off to Methodology / Implementation (out of this section’s ownership).

---

## References（投影片末 / 報告 References）

1. D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” *ICLR*, 2014.  
2. D. J. Rezende, S. Mohamed, and D. Wierstra, “Stochastic Backpropagation and Approximate Inference in Deep Generative Models,” *ICML*, 2014.  
3. D. P. Kingma and M. Welling, “An Introduction to Variational Autoencoders,” *Foundations and Trends in Machine Learning*, 2019.  
4. I. Goodfellow *et al.*, “Generative Adversarial Nets,” *NeurIPS*, 2014.  
5. J. Sohl-Dickstein *et al.*, “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” *ICML*, 2015.  
6. J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” *NeurIPS*, 2020.  
7. Y. Song *et al.*, “Score-Based Generative Modeling through Stochastic Differential Equations,” *ICLR*, 2021.  
8. C. Luo, “Understanding Diffusion Models: A Unified Perspective,” *arXiv:2208.11970*, 2022.  
9. R. Rombach *et al.*, “High-Resolution Image Synthesis with Latent Diffusion Models,” *CVPR*, 2022.  
10. A. Lugmayr *et al.*, “RePaint: Inpainting using Denoising Diffusion Probabilistic Models,” *CVPR*, 2022.

---

## Speaker notes（Literature Review 對稿；不上投影片）

- 開場對齊 supervisor：**不是泛談 GenAI**，而是「LVM = noise+guidance→image → diffusion」。  
- 你的頁面只做 lit review；Motivation / Methodology / Outcome 用 supervisor 架構，不必在本節重寫成你的 aim。  
- VAE→diffusion：用 Luo 的 MHVAE 說法；口頭可講「先會 VAE 才會看懂 diffusion ELBO」。  
- Diffusion 在 lit review **停在 high-level I/O + \(\mathcal{L}_{simple}\)**；算法細節留給 methodology。  
- 一定點到兩個 application：**generation + inpainting**，並各有至少一篇代表文獻。  
- 結尾 R1–R6 交給下一章，不要在 lit review 寫實作步驟。
