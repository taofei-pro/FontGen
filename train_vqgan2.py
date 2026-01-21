import argparse
from pathlib import Path

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
from models.vqgan2.discriminator import PatchGANDiscriminator
from models.vqgan2.loss import VQGAN2Loss
from models.vqgan2.tokenizer import VQGAN2Tokenizer
from utils.hardware.hardware_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQGAN-2")
    parser.add_argument("--target_img_dir", type=str, default=None)
    parser.add_argument("--reference_img_dir", type=str, default=None)
    parser.add_argument("--split_ratios", type=float, nargs=2, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=200)
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
        token_dim=model_config.token_dim,
        codebook_size=model_config.codebook_size,
        commitment_cost=model_config.commitment_cost,
    ).to(device)
    recon_loss_fn = VQGAN2Loss(
        perceptual_weight=training_config.perceptual_weight
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

            tokens, vq_loss = tokenizer.encode(images)
            recon = tokenizer.decode(tokens)
            losses = recon_loss_fn.compute_losses(recon, images)
            recon_loss = losses["total"]
            g_loss = recon_loss + vq_loss

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
            g_optimizer.step()

            if step % 10 == 0:
                print(
                    f"[VQGAN2] step={step} "
                    f"recon={losses['l1'].item():.6f} "
                    f"perc={losses['perceptual'].item():.6f} "
                    f"vq={vq_loss.item():.6f} "
                    f"total={g_loss.item():.6f}"
                )
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_path = Path(training_config.model_save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"tokenizer": tokenizer.state_dict()},
                    save_path,
                )
                print(f"✅ 已保存 VQGAN-2 tokenizer: {save_path}")
                return


if __name__ == "__main__":
    main()
