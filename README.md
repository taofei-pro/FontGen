# FontGen - 字体生成项目说明

本项目是 **HanziGen 的独立升级版**，专注于字体风格学习与生成：

**VQVAE + LDM (Latent Diffusion Model) + 推理**

目标场景：
- 少样本（如 749 字）学习字体风格
- 扩展生成 6763 字全字库
- 支持高质量字体输出

**环境说明**：本项目使用 conda 环境名 `font-gen`（原 `hanzigen` 已改名）。

---

## 🧠 架构概览

```
数据准备
        ↓
VQVAE (向量量化自编码器) → 学习字体的潜在表示
        ↓
LDM (潜在扩散模型) → 在潜在空间中进行扩散生成
        ↓
infer_ldm → 生成字体图像
        ↓
质量评估
```

---

## ✅ 已完成的部分

1) VQVAE 训练
- 已实现 `train_vqvae.py` 训练脚本
- 支持保存 VQVAE 权重
- 模型位于 `models/vqvae/vqvae.py`

2) LDM 训练
- 已实现 `train_ldm.py` 训练脚本
- 支持加载预训练的 VQVAE 权重
- 模型位于 `models/ldm/ldm.py`

3) 推理流程
- 已实现 `infer_ldm.py` 推理脚本
- 支持从训练好的模型生成字体图像

4) 完整 pipeline
- 已提供 `scripts/full_pipeline.sh` 完整训练流程
- 包含数据准备、训练、推理和评估

---

## 🚀 完整训练流程

```bash
# 运行完整训练 pipeline
bash scripts/full_pipeline.sh
```

### 流程详细步骤：

1) **数据准备**
   - `bash scripts/prepare_dataset.sh` - 准备目标字体数据
   - `bash scripts/extract_charset.sh` - 提取字符集
   - `bash scripts/generate_reference.sh` - 生成参考图像

2) **训练 VQVAE**
   - `bash scripts/train_vqvae.sh` - 训练向量量化自编码器

3) **训练 LDM**
   - `bash scripts/train_ldm.sh` - 训练潜在扩散模型

4) **推理生成**
   - `bash scripts/infer_ldm.sh` - 使用训练好的模型生成字体图像

5) **质量评估**
   - `bash scripts/compute_metrics.sh` - 计算生成字体的质量指标

---

## � 单独运行命令

### 数据准备
```bash
bash scripts/prepare_dataset.sh
bash scripts/extract_charset.sh
bash scripts/generate_reference.sh
```

### 训练 VQVAE
```bash
# 使用脚本训练
bash scripts/train_vqvae.sh

# 或直接运行 Python 脚本
python train_vqvae.py --batch_size 1 --num_epochs 100
```

### 训练 LDM
```bash
# 使用脚本训练
bash scripts/train_ldm.sh

# 或直接运行 Python 脚本
python train_ldm.py --vqvae_checkpoint checkpoints/vqvae.pth --batch_size 1 --num_epochs 100
```

### 推理
```bash
# 使用脚本推理
bash scripts/infer_ldm.sh

# 或直接运行 Python 脚本
python infer_ldm.py
```

---

## 📊 训练结果记录

训练与评估结果统一记录在 `RESULTS.md`，每轮优化后追加一行。

---

## 📌 目录结构重点

- `models/vqvae/`：VQVAE 模型实现
- `models/ldm/`：LDM 模型实现
- `datasets/`：数据集相关代码
- `scripts/`：各种脚本文件
  - `full_pipeline.sh`：完整训练流程
  - `train_vqvae.sh`：VQVAE 训练脚本
  - `train_ldm.sh`：LDM 训练脚本
  - `infer_ldm.sh`：推理脚本
- `train_vqvae.py`：VQVAE 训练入口
- `train_ldm.py`：LDM 训练入口
- `infer_ldm.py`：推理入口

---

## 🗺️ 推荐推进路线

1. 运行完整 pipeline 进行端到端测试
2. 根据需要调整 VQVAE 和 LDM 的超参数
3. 评估生成结果并迭代优化
4. 生成完整字库并进行质量检查

---

## 📝 弃用说明

以下脚本已经基本弃用，建议使用新的 VQVAE + LDM 架构相关脚本：

- `scripts/infer.sh` - 已被 `scripts/infer_ldm.sh` 替代
- `scripts/train_vqgan.sh` - 已被 `scripts/train_vqvae.sh` 替代
- `scripts/train_dit.sh` - 已被 `scripts/train_ldm.sh` 替代

---

如需继续开发，直接从这里开始即可，避免上下文丢失。
