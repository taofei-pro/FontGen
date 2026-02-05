import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.vqgan_config import (
    VQGAN2DatasetConfig,
    VQGAN2ModelConfig,
    VQGAN2TrainingConfig,
)
from datasets.dataset_utils import split_dataset
from datasets.image_dataset import PairedGlyphImageDataset
from models.vqgan.discriminator import PatchGANDiscriminator
from models.vqgan.loss import VQGAN2Loss
from models.vqgan.tokenizer import VQGAN2Tokenizer
from utils.hardware.hardware_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQGAN-2")
    parser.add_argument("--target_img_dir", type=str, default=None)
    parser.add_argument("--reference_img_dir", type=str, default=None)
    parser.add_argument("--split_ratios", type=float, nargs=2, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--perceptual_weight", type=float, default=None)
    parser.add_argument("--adversarial_weight", type=float, default=None)
    parser.add_argument("--discriminator_start_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    defaults = VQGAN2DatasetConfig()
    dataset_config = VQGAN2DatasetConfig(
        target_img_dir=args.target_img_dir or defaults.target_img_dir,
        reference_img_dir=args.reference_img_dir or defaults.reference_img_dir,
        split_ratios=tuple(args.split_ratios)
        if args.split_ratios
        else defaults.split_ratios,
        random_seed=args.random_seed or defaults.random_seed,
        batch_size=args.batch_size or defaults.batch_size,
        num_workers=args.num_workers or defaults.num_workers,
    )

    model_config = VQGAN2ModelConfig()
    defaults = VQGAN2TrainingConfig()
    training_config = VQGAN2TrainingConfig(
        batch_size=dataset_config.batch_size,
        model_save_path=args.save_path or defaults.model_save_path,
        perceptual_weight=args.perceptual_weight
        if args.perceptual_weight is not None
        else defaults.perceptual_weight,
        adversarial_weight=args.adversarial_weight
        if args.adversarial_weight is not None
        else defaults.adversarial_weight,
        discriminator_start_steps=args.discriminator_start_steps
        if args.discriminator_start_steps is not None
        else defaults.discriminator_start_steps,
    )

    base_dataset = PairedGlyphImageDataset(
        target_img_dir=dataset_config.target_img_dir,
        reference_img_dir=dataset_config.reference_img_dir,
        use_data_augmentation=False,
    )
    train_dataset, _ = split_dataset(
        dataset=base_dataset,
        split_ratios=dataset_config.split_ratios,
        random_seed=dataset_config.random_seed,
    )

    num_workers = min(2, dataset_config.num_workers)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=1 if num_workers > 0 else None,
        persistent_workers=False,
    )

    tokenizer = VQGAN2Tokenizer(
        in_channels=model_config.input_img_channels,
        latent_dim=model_config.latent_dim,
        base_channels=model_config.base_channels,
        token_dim=model_config.token_dim,
        codebook_size=model_config.codebook_size,
        commitment_cost=model_config.commitment_cost,
        multiscale=model_config.multiscale,
        coarse_downsample=model_config.coarse_downsample,
        coarse_weight=model_config.coarse_weight,
        tanh_output=model_config.tanh_output,
        vq_decay=model_config.vq_decay,
        vq_epsilon=model_config.vq_epsilon,
    ).to(device)
    recon_loss_fn = VQGAN2Loss(
        perceptual_weight=training_config.perceptual_weight,
        foreground_weight=training_config.foreground_weight,
    ).to(device)

    discriminator = None
    if model_config.use_vqgan:
        discriminator = PatchGANDiscriminator(
            in_channels=model_config.input_img_channels
        ).to(device)

    g_optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    d_optimizer = None
    if discriminator is not None:
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=training_config.discriminator_lr,
            weight_decay=training_config.weight_decay,
        )
    
    # 添加学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    g_scheduler = CosineAnnealingWarmRestarts(
        g_optimizer,
        T_0=1000,
        T_mult=2,
        eta_min=1e-6
    )
    d_scheduler = None
    if d_optimizer is not None:
        d_scheduler = CosineAnnealingWarmRestarts(
            d_optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-6
        )

    def d_hinge_loss(real_logits, fake_logits):
        real_loss = torch.relu(1.0 - real_logits).mean()
        fake_loss = torch.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss

    def g_hinge_loss(fake_logits):
        return -fake_logits.mean()

    tokenizer.train()
    if discriminator is not None:
        discriminator.train()

    step = 0
    for epoch in range(training_config.num_epochs):
        for batch in train_loader:
            images = batch["tgt_img"].to(device)

            try:
                tokens, vq_loss = tokenizer.encode(images)
                recon = tokenizer.decode(tokens)
                losses = recon_loss_fn.compute_losses(recon, images)
                recon_loss = losses["total"]
                g_loss = recon_loss + vq_loss
                
                # 检查数值稳定性
                if torch.isnan(g_loss) or torch.isinf(g_loss):
                    print(f"[警告] step={step} 出现异常损失值: g_loss={g_loss.item()}")
                    # 跳过这一步训练
                    continue
                    
            except Exception as e:
                print(f"[错误] step={step} 训练过程中出现异常: {e}")
                # 清理内存并继续
                torch.cuda.empty_cache()
                continue

            use_disc = (
                discriminator is not None
                and d_optimizer is not None
                and step >= training_config.discriminator_start_steps
            )
            adv_loss = torch.zeros_like(g_loss)
            d_loss = torch.zeros_like(g_loss)
            if use_disc:
                # Discriminator step
                d_optimizer.zero_grad()
                real_logits = discriminator(images)
                fake_logits = discriminator(recon.detach())
                d_loss = d_hinge_loss(real_logits, fake_logits)
                d_loss.backward()
                d_optimizer.step()

                # Generator adversarial loss
                adv_logits = discriminator(recon)
                adv_loss = g_hinge_loss(adv_logits)
                g_loss = g_loss + training_config.adversarial_weight * adv_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
            g_optimizer.step()
            g_scheduler.step()
            
            if d_scheduler is not None:
                d_scheduler.step()

            if step % 10 == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[VQGAN] [{current_time}] step={step} "
                    f"recon={losses['l1'].item():.6f} "
                    f"perc={losses['perceptual'].item():.6f} "
                    f"vq={vq_loss.item():.6f} "
                    f"total={g_loss.item():.6f}"
                )
            
            # 定期清理GPU缓存，防止内存泄漏
            if step % 100 == 0:
                torch.cuda.empty_cache()
                
            step += 1
            
            # 每500步保存一次模型，避免长时间训练后丢失进度
            if step % 500 == 0:
                save_path = Path(training_config.model_save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"tokenizer": tokenizer.state_dict(), "step": step},
                    save_path,
                )
                print(f"✅ 已保存 VQGAN-2 tokenizer (step {step}): {save_path}")
                
            if args.max_steps and step >= args.max_steps:
                save_path = Path(training_config.model_save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"tokenizer": tokenizer.state_dict(), "step": step},
                    save_path,
                )
                print(f"✅ 已保存最终 VQGAN-2 tokenizer: {save_path}")
                return


if __name__ == "__main__":
    main()
