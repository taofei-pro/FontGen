# FontGen - 最强架构项目说明

本项目是 **HanziGen 的独立升级版**，聚焦最强架构路线：

**VQGAN‑2 + DiT + 结构条件 + 超分 + 矢量化**

目标场景：
- 少样本（如 749 字）学习字体风格
- 扩展生成 6763 字全字库
- 支持高分辨率输出与字体包生产

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

---

## 🔧 仍需完成的部分

1) VQGAN‑2 完整训练
- 加入感知损失与判别器稳定训练
- 导出 tokenizer checkpoint

2) DiT 采样/推理
- 加入 DDIM/DPM 等采样器
- 实现完整推理与生成流程

3) 超分链路
- 接入 SR 模型（Real‑ESRGAN / SwinIR）
- 实现 patch/tiling 推理

4) 矢量化与字体包输出
- Potrace → SVG
- FontForge → TTF/OTF

---

## 🚀 当前可用训练命令（最小验证）

```bash
# VQGAN‑2 tokenizer 训练（最小闭环）
python train_vqgan2.py --max_steps 20

# DiT 训练（最小闭环，默认结构条件）
python train_dit.py --max_steps 20

# DiT 使用预训练 tokenizer
python train_dit.py --vqgan2_ckpt checkpoints/vqgan2.pth --max_steps 20
```

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
