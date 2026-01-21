# FontGen 训练结果记录

> 说明：记录每次优化后的训练与评估结果，便于对比。

| 轮次 | 数据尺寸 | VQGAN 权重 | DiT 步数 | SR 步数 | 采样配置 | PSNR | SSIM | LPIPS | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128x128 | checkpoints/vqgan.pth | 30000 | 10000 | dpmpp_3m + karras, cfg_rescale=0.4, x0_clip=1.5 | 6.7480 | 0.0794 | 0.7264 | 当前结果（偏模糊） |
| 2 | 256x256 | checkpoints/vqgan.pth | 50000 | 20000 | dpmpp_3m + karras, cfg_rescale=0.3, x0_clip=2.0 | - | - | - | 待重训与评估 |