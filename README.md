# FontGen - 字体生成项目说明

本项目是 **HanziGen 的独立升级版**，专注于字体风格学习与生成：

**VQVAE + LDM (Latent Diffusion Model) + 推理**

目标场景：
- 少样本（如 749 字）学习字体风格
- 扩展生成 6763 字全字库
- 支持高质量字体输出

学习对象：
https://github.com/wangwenho/HanziGen
https://github.com/kaonashi-tyc/zi2zi-JiT

**环境说明**：本项目使用 conda 环境名 `font-gen`。

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

### 架构详细原理

#### 1. 数据准备
**作用**：为模型训练提供高质量的字体图像数据。

**详细过程**：
- **目标字体数据准备**：从目标字体文件中提取字形图像
- **字符集提取**：确定需要学习和生成的字符集合
- **参考图像生成**：为每个目标字符生成对应的参考风格图像

**技术要点**：
- 确保字体图像的一致性（大小、分辨率、对比度）
- 建立目标字体与参考字体之间的对应关系
- 为后续模型训练提供标准化的输入数据

#### 2. VQVAE (向量量化自编码器)
**作用**：学习字体的潜在表示，将高维图像压缩为低维离散表示。

**工作原理**：
1. **编码过程**：通过编码器网络将输入的字体图像压缩为低维潜在表示
2. **向量量化**：将连续的潜在表示映射到离散的码本空间，学习字体的基本视觉单元
3. **解码过程**：通过解码器网络将量化后的潜在表示重建为字体图像

**技术要点**：
- **码本学习**：通过 commitment loss 确保潜在表示与码本向量的接近度
- **离散表示**：将字体特征编码为离散的"视觉词汇"，便于后续扩散模型处理
- **压缩效率**：大幅减少数据维度，加速后续扩散模型的训练

#### 3. LDM (潜在扩散模型)
**作用**：在潜在空间中学习字体风格的分布，实现高质量的字体生成。

**工作原理**：
1. **噪声添加**：在潜在空间中逐步添加噪声，将清晰的潜在表示转变为随机噪声
2. **噪声预测**：训练 UNet 网络预测添加的噪声
3. **逆向扩散**：在推理时，从随机噪声开始，逐步移除预测的噪声，生成新的潜在表示

**技术要点**：
- **条件生成**：利用参考字体的潜在表示作为条件，引导生成过程
- **潜在空间操作**：在压缩的潜在空间中进行扩散，大幅减少计算量
- **高质量生成**：通过逐步降噪过程，生成细节丰富的字体图像

#### 4. 推理生成 (infer_ldm)
**作用**：使用训练好的模型生成新的字体图像。

**工作原理**：
1. **编码参考图像**：将参考字体图像编码为潜在表示
2. **扩散生成**：在潜在空间中从随机噪声开始，根据参考潜在表示生成目标风格的潜在表示
3. **解码生成**：将生成的潜在表示解码为最终的字体图像

**技术要点**：
- **条件控制**：通过参考图像控制生成字体的风格
- **采样策略**：使用 DDIM 等高效采样方法加速生成过程
- **批量生成**：支持一次性生成大量字体图像

#### 5. 质量评估
**作用**：评估生成字体的质量，指导模型优化。

**评估指标**：
- **SSIM (结构相似性指数)**：衡量生成图像与目标图像的结构相似程度，**数值越大越好**（范围：0-1）
- **PSNR (峰值信噪比)**：衡量生成图像的保真度，**数值越大越好**（单位：dB）
- **FID (弗雷歇 inception 距离)**：衡量生成字体分布与真实字体分布的差异，**数值越小越好**（理想值：0）

**技术要点**：
- 综合多个指标全面评估生成质量
- 建立标准化的评估流程，确保结果的可比较性
- 根据评估结果调整模型超参数，持续优化生成效果

### 架构优势

1. **少样本学习**：通过 VQVAE 学习字体的基本视觉单元，只需少量样本即可掌握字体风格
2. **高质量生成**：LDM 在潜在空间中进行精细的噪声预测，生成的字体细节丰富
3. **高效训练**：在压缩的潜在空间中训练扩散模型，大幅减少计算资源需求
4. **灵活控制**：通过参考图像可以灵活控制生成字体的风格
5. **可扩展性**：易于扩展到不同语言和字体风格

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
