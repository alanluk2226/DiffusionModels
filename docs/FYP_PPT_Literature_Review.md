# FYP PPT — (3) Literature Review（完整稿）

> 專題：From-Scratch Learning of VAEs and Denoising Diffusion Models in PyTorch  
> 本檔為 PPT **Literature Review** 投影片文案（可直接貼入簡報）。  
> 對應 Initial Report Chapter 2；引用 ≥6 篇學術文獻。

---

# Slide A — Literature Review Overview

**Chapter outline**

1. Generative modelling landscape  
2. Variational Autoencoders (VAE)  
3. From hierarchical VAE to diffusion  
4. Denoising Diffusion Probabilistic Models (DDPM)  
5. Libraries / tutorials vs first-principles learning  
6. Domain requirements & proposed gap  

**Core claim of this review**

VAE provides the **ELBO + latent-variable** foundation; diffusion is best understood as a **deep Markovian VAE** with a fixed Gaussian noising encoder and a learned denoiser [Luo, 2022].

---

# Slide B — Generative Modelling Landscape

| Family | Core idea | Strength | Limitation for *this* FYP |
|--------|-----------|----------|---------------------------|
| Autoencoder | Compress → reconstruct | Simple; good features | Latent not a proper prior → cannot sample reliably |
| **VAE** [Kingma & Welling, 2014] | Maximise ELBO = recon − KL | Tractable likelihood bound; stable training | Often blurry samples; maths often skipped in tutorials |
| GAN [Goodfellow et al., 2014] | Adversarial min–max | Sharp images | Unstable; no explicit ELBO story this project needs |
| **DDPM** [Ho et al., 2020] | Learn reverse of noise chain | High sample quality; clear sampling path | Heavy; easy to treat as black-box U-Net |
| Score / SDE models [Song et al., 2021] | Estimate score ∇ log *p* | Unifies continuous diffusion | Extra abstraction beyond FYP scope |
| High-level libs (e.g. Diffusers) | Call pretrained pipelines | Fast demos | Hide equations this FYP must master |

**Lesson learned:** For equation-level mastery, prefer models with an **explicit likelihood / ELBO story** (VAE, DDPM) over pure adversarial training.

---

# Slide C — VAE: Problem & Formulation

**Problem VAE solves**

Marginal likelihood is intractable for flexible latent-variable models:

\[
p_\theta(x)=\int p_\theta(x|z)\,p(z)\,dz
\]

**Solution [Kingma & Welling, 2014; Rezende et al., 2014]**

Introduce approximate posterior \(q_\phi(z|x)\) and optimise the **Evidence Lower Bound (ELBO)**:

\[
\log p_\theta(x)
=
\underbrace{\mathbb{E}_{q_\phi(z|x)}\!\Big[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\Big]}_{\mathrm{ELBO}(\theta,\phi;x)}
+
D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p_\theta(z|x)\big)
\]

Since KL ≥ 0 ⇒ ELBO ≤ log evidence. Maximising ELBO raises a lower bound on log-likelihood *and* shrinks the gap to the true posterior.

---

# Slide D — VAE: ELBO Decomposition & Reparameterisation

**Practical training objective**

\[
\mathrm{ELBO}
=
\underbrace{\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big]}_{\text{reconstruction}}
-
\underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{prior matching}}
\]

Typical choices: \(q_\phi(z|x)=\mathcal{N}(\mu_\phi(x),\sigma_\phi^2(x)I)\), \(p(z)=\mathcal{N}(0,I)\).

Closed-form KL (diagonal Gaussian, per dim \(i\)):

\[
D_{\mathrm{KL}}=\tfrac12\sum_i\big(\mu_i^2+\sigma_i^2-1-\log\sigma_i^2\big)
\]

**Reparameterisation trick** (enables backprop through sampling):

\[
z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I)
\]

**Why VAE matters for generative modelling**

- Continuous, regularised latent space → sample by \(z\sim p(z)\) then decode  
- Interpolation / latent arithmetic when space is structured  
- Conceptual bridge to diffusion [Luo, 2022]

**Known limitation:** pixel-space VAEs often produce **blurry** samples vs GANs / diffusion [Kingma & Welling, 2019].

---

# Slide E — Bridge: Hierarchical VAE → Diffusion

**Markovian Hierarchical VAE (MHVAE)** [Luo, 2022]

\[
p(x,z_{1:T})=p(z_T)\,p_\theta(x|z_1)\prod_{t=2}^{T}p_\theta(z_{t-1}|z_t)
\]

\[
q(z_{1:T}|x)=q(z_1|x)\prod_{t=2}^{T}q(z_t|z_{t-1})
\]

**Diffusion as a special MHVAE when:**

1. Latent dimension = data dimension (\(z_t \equiv x_t\))  
2. Forward (encoder) transitions are **fixed** linear-Gaussian noise  
3. \(x_T\sim\mathcal{N}(0,I)\) (pure noise prior)  
4. Only the **reverse denoiser** \(p_\theta(x_{t-1}|x_t)\) is learned  

→ Studying VAE first is not optional; it is the **premise** of understanding diffusion ELBOs.

---

# Slide F — Diffusion Origins & DDPM

**Historical line**

| Work | Contribution |
|------|----------------|
| Sohl-Dickstein et al., 2015 | Diffusion / nonequilibrium thermodynamics idea for generative models |
| Ho, Jain & Abbeel, 2020 (**DDPM**) | Practical high-quality image synthesis; simplified noise-prediction loss |
| Song et al., 2021 | Score-based / SDE unified continuous view |
| Luo, 2022 | Tutorial unifying VAE ↔ diffusion equations |
| Rombach et al., 2022 (LDM) | Latent diffusion (VAE encoder + diffusion in latent space) — SOTA path, out of FYP depth |

**DDPM core message [Ho et al., 2020]**  
Destroy data with a fixed noise schedule; train a network to reverse the process; sampling starts from Gaussian noise.

---

# Slide G — DDPM Forward Process (must-know equations)

**One-step noising (variance-preserving):**

\[
q(x_t|x_{t-1})=\mathcal{N}\!\big(x_t;\sqrt{\alpha_t}\,x_{t-1},\,(1-\alpha_t)I\big)
\]

Define \(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\). **Closed form** (jump to any \(t\)):

\[
q(x_t|x_0)=\mathcal{N}\!\big(x_t;\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\big)
\]

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I)
\]

**Implication for code:** training can sample \(t\) uniformly and form \(x_t\) in one line — no need to simulate the whole chain.

---

# Slide H — DDPM Reverse Process & True Posterior

**Generative (reverse) model:**

\[
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t),\quad p(x_T)=\mathcal{N}(0,I)
\]

**True reverse posterior given \(x_0\)** (training target):

\[
q(x_{t-1}|x_t,x_0)=\mathcal{N}\!\big(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\big)
\]

with \(\beta_t=1-\alpha_t\):

\[
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
\]

\[
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
\]

Training matches \(p_\theta(x_{t-1}|x_t)\) to this Gaussian — again an **ELBO / KL matching** view [Luo, 2022].

---

# Slide I — Simplified Noise-Prediction Loss

Express \(\tilde\mu\) via predicted noise \(\varepsilon_\theta(x_t,t)\). Optimising the KL between Gaussians reduces (up to weighting) to the **practical DDPM objective** [Ho et al., 2020]:

\[
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{t,\,x_0,\,\varepsilon}
\Big[\big\|\varepsilon-\varepsilon_\theta(x_t,t)\big\|_2^2\Big]
\]

where \(x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon\).

**Sampling (ancestral):** \(x_T\sim\mathcal{N}(0,I)\); for \(t=T,\ldots,1\) predict \(\varepsilon_\theta\), form mean, add noise if \(t>1\); output \(x_0\).

**One-sentence mental model**

> Diffusion = deep Markovian VAE with fixed Gaussian encoder + learned denoiser; ELBO → “predict the noise that was added.”

---

# Slide J — Related Solutions: Lessons → Domain Requirements

| Source | + Lesson | − Lesson |
|--------|----------|----------|
| Classic AE | Reconstruction is intuitive | No sampling prior |
| VAE papers / surveys | ELBO + reparam are teachable | Blurry; tutorials skip derivation |
| GAN literature | Sharp samples possible | Unstable; weak likelihood story |
| DDPM / Luo tutorial | Clear maths–sampling path | Slow sampling; heavy U-Net |
| Diffusers / Kaggle notebooks | Fast visual demos | Equations buried; copy-paste risk |
| Latent Diffusion [Rombach et al., 2022] | Production-quality path | Too large for first-principles FYP |

**Domain requirements derived from the review**

| ID | Requirement |
|----|-------------|
| R1 | Tractable training objective (ELBO / \(\mathcal{L}_{\mathrm{simple}}\)) |
| R2 | Explicit sampling path (decode \(z\) / reverse chain) |
| R3 | Transparent **maths ↔ code** mapping |
| R4 | Runnable on limited student hardware (MNIST-scale) |
| R5 | Cover **both** VAE and DDPM (VAE as premise) |

---

# Slide K — Proposed Solution vs Existing Work

| Criterion | Typical tutorial / library | SOTA (e.g. LDM) | **This FYP** |
|-----------|----------------------------|-----------------|--------------|
| Focus | Pretty samples | Production quality | Equations + correct minimal code |
| Maths | Partial / skipped | Assumed known | Derive & state key equations |
| Scope | VAE *or* diffusion | Full latent diffusion stack | **Both** VAE then DDPM |
| Stack | High-level APIs | Large pretrained systems | From-scratch PyTorch |
| Success metric | FID / demos | FID, CLIP, etc. | Working models + clear write-up |
| Hardware | Cloud GPU assumed | Multi-GPU | CPU / single GPU friendly |

**Honest gap:** sample quality and speed will **not** match Stable Diffusion.  
**Value:** first-principles clarity — every loss term maps to a few lines of code.

---

# Slide L — Literature Review Summary (take-home)

1. **VAE** = learn \(q_\phi(z|x)\) + \(p_\theta(x|z)\); train with ELBO (recon − KL) + reparameterisation.  
2. **Diffusion** ≈ deep Markovian VAE with **fixed** Gaussian noising and a **learned** denoiser.  
3. Practical DDPM objective: predict noise \(\varepsilon\) added to \(x_0\).  
4. Prior art either **hides maths** (libraries) or **targets SOTA** (too heavy); this project fills the **equation ↔ from-scratch code** gap on a small dataset.  
5. Requirements R1–R5 guide methodology (next section).

---

## References（投影片末 / 報告 References）

1. D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” *ICLR*, 2014.  
2. D. J. Rezende, S. Mohamed, and D. Wierstra, “Stochastic Backpropagation and Approximate Inference in Deep Generative Models,” *ICML*, 2014.  
3. D. P. Kingma and M. Welling, “An Introduction to Variational Autoencoders,” *Foundations and Trends in Machine Learning*, 2019.  
4. I. Goodfellow *et al.*, “Generative Adversarial Nets,” *NeurIPS*, 2014.  
5. J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” *ICML*, 2015.  
6. J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” *NeurIPS*, 2020.  
7. Y. Song *et al.*, “Score-Based Generative Modeling through Stochastic Differential Equations,” *ICLR*, 2021.  
8. C. Luo, “Understanding Diffusion Models: A Unified Perspective,” *arXiv:2208.11970*, 2022.  
9. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-Resolution Image Synthesis with Latent Diffusion Models,” *CVPR*, 2022.

*(≥6 academic citations satisfied; 9 listed for Initial Report.)*

---

## Speaker notes（對稿用，不上投影片）

- 開場一句：Lit review 不是列論文，是**抽出 R1–R5**，用來評斷既有方案、導向本專題。  
- 強調 **VAE → MHVAE → DDPM** 一條線，呼應導師「VAE 是理解 diffusion 的前提」。  
- 方程頁只講「這項在程式哪裡」：ELBO→loss；\(x_t\) closed form→`q_sample`；\(\mathcal{L}_{simple}\)→MSE。  
- 對比表最後一欄必須誠實：不做 SOTA，做 first principles。  
- 若時間緊：B + D + E + I + K 五頁最關鍵。
