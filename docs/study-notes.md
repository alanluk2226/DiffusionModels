# FYP Study Notes: VAE → Diffusion → Proposal Writing

Last updated: 2026-09-28
Status: Documentation study phase (no coding yet per user request)

## Learning path (user-assigned)

1. PyTorch refresher (optional): Aladdin Persson playlist vids 1–18
2. VAE foundation: playlist vids 1–3 (+ optional 4); intuitive VAE video `qJeaCHQ1k2w`
3. Diffusion: Calvin Luo arXiv `2208.11970` + talk `bGh_uBbj_Po`
4. Later (after mastery): code VAE from scratch; code DDPM from scratch (Kaggle + Outlier video as refs)
5. Proposal writing: Topic 4 Part1 + Part2 PDFs (HKMU COMP FYP)

## Project context note

Workspace `EverydayLens` is a separate portfolio classifier demo (Fashion-MNIST → ONNX → FastAPI → Next.js). FYP learning track here is generative modeling (VAE/Diffusion), not EverydayLens implementation.

---

# Part A — VAE (must-know equations)

## 1. Latent-variable generative goal

- Observe data \(x \sim p_{\text{data}}(x)\).
- Introduce latent \(z\); joint \(p(x,z)\).
- Marginal likelihood (intractable for complex models):
  \[
  p(x) = \int p(x,z)\,dz = \frac{p(x,z)}{p(z|x)}
  \]

## 2. Evidence Lower Bound (ELBO)

With variational encoder \(q_\phi(z|x)\):

\[
\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}\!\left[\log\frac{p(x,z)}{q_\phi(z|x)}\right]}_{\text{ELBO}} + D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p(z|x)\big)
\]

- KL ≥ 0 ⇒ ELBO ≤ log evidence.
- Maximizing ELBO w.r.t. \(\phi\) shrinks the gap to the true posterior (proxy for matching \(p(z|x)\)).

Jensen form:
\[
\log p(x) = \log\mathbb{E}_{q}\!\left[\frac{p(x,z)}{q(z|x)}\right] \ge \mathbb{E}_{q}\!\left[\log\frac{p(x,z)}{q(z|x)}\right]
\]

## 3. VAE ELBO decomposition (reconstruction − prior KL)

\[
\mathrm{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x)}\!\big[\log p_\theta(x|z)\big]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{prior matching}}
\]

Typical choices:
\[
q_\phi(z|x)=\mathcal{N}(z;\mu_\phi(x),\sigma_\phi^2(x)I),\quad p(z)=\mathcal{N}(0,I)
\]

Closed-form KL for diagonal Gaussians (per dim \(i\)):
\[
D_{\mathrm{KL}} = \tfrac12\sum_i\big(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2\big)
\]

## 4. Reparameterization trick

\[
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
\]

Enables backprop through stochastic sampling.

## 5. Why VAE matters for generative modeling (intuition)

- Learns a continuous, regularized latent space (not just a point code).
- Sampling: \(z\sim p(z)\) → decode \(p_\theta(x|z)\).
- Latent arithmetic / interpolation possible when space is semantically structured.
- Foundation for thinking of diffusion as a deep / Markovian hierarchical VAE with fixed Gaussian encoders.

## 6. Markovian Hierarchical VAE (bridge to diffusion)

\[
p(x,z_{1:T}) = p(z_T)\,p_\theta(x|z_1)\prod_{t=2}^{T}p_\theta(z_{t-1}|z_t)
\]
\[
q_\phi(z_{1:T}|x) = q_\phi(z_1|x)\prod_{t=2}^{T}q_\phi(z_t|z_{t-1})
\]

Diffusion = MHVAE with: (i) latent dim = data dim, (ii) fixed linear-Gaussian forward, (iii) \(x_T\sim\mathcal{N}(0,I)\).

---

# Part B — Diffusion / DDPM (Calvin Luo unified view)

## 1. Forward (noising) process — fixed encoder

Variance-preserving form:
\[
q(x_t|x_{t-1}) = \mathcal{N}\!\big(x_t;\sqrt{\alpha_t}\,x_{t-1},\,(1-\alpha_t)I\big)
\]

Define \(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\). Closed form (jump to any \(t\)):
\[
q(x_t|x_0)=\mathcal{N}\!\big(x_t;\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\big)
\]
\[
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)
\]

## 2. Reverse (denoising) generative model

\[
p(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t),\quad p(x_T)=\mathcal{N}(0,I)
\]

Learn \(p_\theta(x_{t-1}|x_t)\) (usually Gaussian).

## 3. True posterior of one reverse step (given \(x_0\))

\[
q(x_{t-1}|x_t,x_0)=\mathcal{N}\!\big(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\big)
\]

with (standard DDPM identities; \(\beta_t=1-\alpha_t\)):
\[
\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
\]
\[
\tilde\beta_t=\frac{(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\beta_t
\]

## 4. ELBO views (two equivalent stories)

### Consistency / matching view (higher variance MC)
- Reconstruction: \(\mathbb{E}[\log p_\theta(x_0|x_1)]\)
- Prior matching at \(T\) (≈0 if schedule good)
- Consistency: match forward \(q(x_t|x_{t-1})\) vs reverse \(p_\theta(x_t|x_{t+1})\)

### Denoising matching view (preferred, lower variance)
\[
\mathrm{ELBO} \approx \underbrace{\mathbb{E}[\log p_\theta(x_0|x_1)]}_{\text{recon}} - D_{\mathrm{KL}}(q(x_T|x_0)\|p(x_T)) - \sum_{t=2}^{T}\mathbb{E}_{q(x_t|x_0)}\!\big[D_{\mathrm{KL}}(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t))\big]
\]

Train reverse step to match the true posterior \(q(x_{t-1}|x_t,x_0)\).

## 5. Parameterizations → simplified noise-prediction loss

Express \(\tilde\mu\) via predicted noise \(\epsilon_\theta(x_t,t)\). Optimizing KL between Gaussians reduces (up to weighting) to:

\[
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\Big[\|\epsilon - \epsilon_\theta(x_t,t)\|_2^2\Big]
\]

where \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\).

This is the practical DDPM training objective used in from-scratch implementations.

## 6. Sampling (ancestral / DDPM)

1. Sample \(x_T\sim\mathcal{N}(0,I)\)
2. For \(t=T,\ldots,1\): predict \(\epsilon_\theta(x_t,t)\); form mean of \(p_\theta(x_{t-1}|x_t)\); add noise if \(t>1\)
3. Output \(x_0\)

## 7. Mental model (one sentence)

Diffusion is a deep Markovian VAE whose encoder is a fixed Gaussian noise schedule and whose decoder is a learned denoiser; training maximizes an ELBO that becomes “predict the noise that was added.”

---

# Part C — HKMU FYP Proposal structure (from Topic 4 Part1 + Part2)

Word limit ≈ 3000.

## Required skeleton

1. **Cover**: Name, HKMU ID, Programme, Date, Supervisor, Project Title (precise one-liner; not nomination-topic copy)
2. **TOC**
3. **Ch1 Problem Definition**
   - Introduction (context → why this problem)
   - Aim (What / direction)
   - Objectives (5–7 measurable How steps; action verbs: Design / Implement / Evaluate / Analyze / Collect / Setup; ≥1 evaluation via prototype)
   - Value proposition (immediate benefits + ripple effects)
4. **Ch2 Literature Review**
   - Existing solutions: lessons learned (+/−), domain requirements
   - Technologies: what/how, fit to requirements, evolution / misuse
   - Highlight proposed solution vs existing (honest comparison)
   - ≥6 academic citations (journals/proceedings) for initial report
5. **Ch3 Preliminary Methodology**
   - Overview of Your How (tie objectives together)
   - Requirements + key technologies + technical gap
   - Architecture: ≥2 diagrams (component + data-flow); describe components, connections, I/O
6. **References** (IEEE / APA / Harvard — consistent)
7. **Appendix A Project Plan** (tasks, weeks, roles)
8. Group only: roles/responsibilities + meeting minutes

## Writing discipline drilled in slides

- Aim = problem+direction; Objectives = measurable divide-and-conquer modules
- Lit review ends in requirements used to judge prior work
- Methodology = methods for solve / implement / design / manage / test / research
- Cite honestly; quotes only when necessary

---

# Part D — Implementation checklist (deferred until user asks)

## VAE from scratch (self-sourced refs)

- [ ] Encoder → μ, logσ²
- [ ] Reparam + decode
- [ ] Loss = recon (BCE/MSE) + β·KL
- [ ] Train on MNIST/CIFAR; latent walk / sample grid

## DDPM from scratch (refs: Kaggle Vikram Sandu; YouTube vu6eKteJWew)

- [ ] β / α / ᾱ schedule buffers
- [ ] `q_sample(x0,t,eps)` closed form
- [ ] U-Net ε_θ(x_t,t) with timestep embedding
- [ ] L_simple MSE
- [ ] `p_sample` loop
- [ ] Visualize noise trajectory + final samples

---

# Open items / next study actions

1. Watch VAE vids 1–3 + intuitive video (equations ↔ intuition)
2. Watch Luo talk while tracing paper §§ VDM / closed-form / ε-param
3. Self-quiz: derive ELBO both ways; derive \(x_t\) closed form; derive why L_simple ≈ KL matching
4. Only after that: open empty repo and code VAE then DDPM from scratch
