# FontGen - 最强架构项目说明

本项目是 **HanziGen 的独立升级版**，聚焦最强架构路线：

**VQGAN‑2 + DiT + 结构条件 + 超分 + 矢量化**

目标场景：
- 少样本（如 749 字）学习字体风格
- 扩展生成 6763 字全字库
- 支持高分辨率输出与字体包生产

**环境说明**：本项目使用 conda 环境名 `font-gen`（原 `hanzigen` 已改名）。

---

## 🧠 最强架构概览

```
结构条件(偏旁/边缘/骨架)
        ↓
VQGAN‑2 Tokenizer (离散 token 空间)
        ↓
DiT (扩散 Transformer 去噪)
        ↓
超分模型 (SR: 512→1024+)
        ↓
矢量化 (Potrace/FontForge)
```

---

## ✅ 已完成的部分

1) 结构条件管线
- 已接入 component mask / edge map / skeleton
- `datasets/structure_dataset.py` 可输出结构条件张量

2) VQGAN‑2 tokenizer
- 已加入量化/码本
- 支持 `train_vqgan2.py` 训练并保存权重

3) DiT 训练流程
- 已接入 VQGAN‑2 tokenizer
- 已升级为标准扩散训练（加噪/去噪）
- 支持加载训练好的 tokenizer 权重
- 已替换为 Transformer DiT 结构（基础版）

---

## 🔧 仍需完成的部分

1) VQGAN‑2 完整训练
- ✅ 已加入 LPIPS 感知损失与判别器稳定训练（hinge + warmup）
- ✅ 支持导出 tokenizer checkpoint

2) DiT 采样/推理
- ✅ 已提供 `infer.py` 基础采样流程（DDIM）
- ✅ 支持 DDPM 采样
- ✅ 支持 DPM++ 2M（`--sampler dpmpp_2m`）
- ✅ 支持 DPM++ 2S（`--sampler dpmpp_2s`）
- ✅ 支持 DPM++ 3M（`--sampler dpmpp_3m`）
- ✅ 支持 Karras 噪声调度（`--schedule karras`）
- ✅ 2S 中点步使用 lambda→sigma→t 映射（更稳定）
- ✅ 支持 CFG Rescale（`--cfg_rescale`）与 x0 clip（`--x0_clip`）
- ✅ DPM++ 多步采样加入历史截断（更稳定）
- ✅ 末步直接回落到 x0（更稳，含 DDIM）
- TODO: DPM-Solver++ 更完整策略与噪声调度改进

3) 超分链路
- ✅ 已提供 `train_sr.py` 基础 SR 训练与 `tile_infer` 推理
- ✅ 增加 EDSR 结构（`--model_name edsr`）
- ✅ 允许外部 Real‑ESRGAN / SwinIR（需 TorchScript 权重，`--sr_model realesrgan|swinir`）
- TODO: 接入 Real‑ESRGAN / SwinIR 等更强模型

4) 矢量化与字体包输出
- ✅ `convert_to_svg.py`（Potrace → SVG）
- ✅ `fontforge_pipeline.py`（FontForge → TTF/OTF，可选）

---

## 🚀 当前可用训练命令（最小验证）

```bash
# 0) 字体覆盖率分析与数据准备（以 fonts/M8.ttf 为目标）
bash scripts/analyze_font.sh
bash scripts/prepare_dataset.sh
bash scripts/extract_charset.sh

# VQGAN‑2 tokenizer 训练（最小闭环）
python train_vqgan2.py --max_steps 20
# 可选：--perceptual_weight 0.4 --adversarial_weight 0.1 --discriminator_start_steps 500

# DiT 训练（最小闭环，默认结构条件）
python train_dit.py --max_steps 20

# DiT 使用预训练 tokenizer
python train_dit.py --vqgan2_ckpt checkpoints/vqgan.pth --max_steps 20
```

---

## 🧪 FontGen 推理与向量化（最小闭环）

```bash
# 1) 推理生成
bash scripts/infer.sh

# 2) Potrace 矢量化
bash scripts/convert_to_svg.sh

# 3) FontForge 打包字体（需系统安装 fontforge 命令）
bash scripts/svg_to_font.sh
```

---

## 🧰 训练流程与调参建议（准备训练）

```bash
# 1) 数据准备（M8 作为目标字）
# 注意：已在 tokenizer 中加入下采样，默认使用 128x128
bash scripts/analyze_font.sh
bash scripts/prepare_dataset.sh
bash scripts/extract_charset.sh

# 2) VQGAN‑2 训练（建议先跑 5k~20k steps）
bash scripts/train_vqgan.sh

# 3) DiT 训练（加载 tokenizer）
bash scripts/train_dit.sh

# 4) 超分训练（EDSR）
bash scripts/train_sr.sh

# 5) 推理（已启用 SR）
bash scripts/infer.sh

# 6) 矢量化与字体输出
bash scripts/convert_to_svg.sh
bash scripts/svg_to_font.sh

# 7) 相似度评估（可选）
bash scripts/compute_metrics.sh
```

调参建议（起点）：
- 采样：`dpmpp_3m` + `karras`，`cfg_rescale=0.4`，`x0_clip=1.5`
- 若显存紧：降低 `batch_size`、`sampling_steps`，或关闭 SR
- 若对抗不稳定：调大 `discriminator_start_steps`（如 1000）

---

## ✅ 完整性检查（当前状态）

- FontGen 主流程已完整：数据准备 → VQGAN‑2 → DiT → 推理 → SR → 矢量化 → 字体输出
- 旧版 LDM 相关脚本已移除，避免误用

---

## 📌 目录结构重点

- `configs/`：VQGAN‑2 / DiT / SR / Structure 配置
- `models/vqgan2/`：VQGAN‑2 tokenizer + 判别器
- `models/dit/`：DiT + Scheduler
- `datasets/structure_dataset.py`：结构条件数据集
- `train_vqgan2.py` / `train_dit.py`：训练入口

---

## 🗺️ 推荐推进路线

1. 完成 VQGAN‑2 训练（高质量 tokenizer）
2. 完成 DiT 采样推理
3. 接入超分与矢量化
4. 生成全字库并导出字体包

---

如需继续开发，直接从这里开始即可，避免上下文丢失。
