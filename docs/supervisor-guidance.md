# Supervisor PPT guidance（摘錄整理）

來源：supervisor 投影片草稿（2026）。本專題你負責的部分：**Literature Review only**。

## 1. Project motivation（整份 PPT 的 framing，非你的章節正文）

- Generative AI
  - LLM（large language model）
  - LVM（large vision model）：input = noise + guidance，output = image → converge to **diffusion model**
  - Multi-modal（LLM + LVM + audio + …）
- Goal direction: understand generative LVM in depth
- Application（2 tasks）
  - Image generation
  - Image inpainting

## 2. Methodology（他人／後續章節；lit review 需鋪墊至此）

- VAE：結合學習與算法認知做 slides
- Diffusion：從 VAE 推廣；說明 VAE 與 diffusion 的關係；algorithm 的 input/output 與 high-level procedure（細節可後談）
- Implementation：PyTorch

## 3. Intended outcome（他人／後續章節）

- MNIST：toy dataset，可完成理論學習（train + test）
- Real image（CIFAR-10 32×32、ImageNet 256×256）：使用 pre-trained，主要做 Application
- Schedule / others：待補

## Implication for Literature Review

Lit review 應回答：

1. Generative AI 裡 LVM 為何走向 diffusion？  
2. VAE 為何是理解 diffusion 的基礎／如何銜接？  
3. Diffusion 在 **image generation** 與 **inpainting** 上既有做法與限制？  
4. 從既有方案抽出 domain requirements，銜接本專題「先理論（MNIST）再應用（pretrained）」路徑。
