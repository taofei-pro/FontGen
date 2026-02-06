import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.dit_config import DiTDatasetConfig, DiTModelConfig, DiTTrainingConfig
from configs.structure_config import StructureConfig
from configs.vqgan_config import VQGAN2ModelConfig
from datasets.dataset_utils import split_dataset
from datasets.image_dataset import PairedGlyphImageDataset
from datasets.structure_dataset import StructureConditionDataset
from models.dit.condition_encoder import ConditionEncoder
from models.dit.dit_model import DiTModel
from models.dit.scheduler import DiffusionScheduler
from models.vqgan.tokenizer import VQGAN2Tokenizer
from utils.hardware.hardware_utils import print_model_params, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DiT")
    parser.add_argument("--target_img_dir", type=str, default=None)
    parser.add_argument("--reference_img_dir", type=str, default=None)
    parser.add_argument("--split_ratios", type=float, nargs=2, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default=None,
        help="Path to pretrained VQGAN-2 tokenizer checkpoint",
    )
    parser.add_argument("--use_component_mask", action="store_true")
    parser.add_argument("--use_edge_map", action="store_true")
    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_ids", action="store_true")
    parser.add_argument("--save_path", type=str, default="checkpoints/dit.pth")
    return parser.parse_args()


def build_structure_config(args: argparse.Namespace) -> StructureConfig:
    defaults = StructureConfig()
    flags_provided = (
        args.use_component_mask
        or args.use_edge_map
        or args.use_skeleton
        or args.use_ids
    )
    if not flags_provided:
        return defaults
    return StructureConfig(
        use_ids=args.use_ids,
        use_component_mask=args.use_component_mask,
        use_skeleton=args.use_skeleton,
        use_edge_map=args.use_edge_map,
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    defaults = DiTDatasetConfig()
    dataset_config = DiTDatasetConfig(
        target_img_dir=args.target_img_dir or defaults.target_img_dir,
        reference_img_dir=args.reference_img_dir or defaults.reference_img_dir,
        split_ratios=tuple(args.split_ratios)
        if args.split_ratios
        else defaults.split_ratios,
        random_seed=args.random_seed or defaults.random_seed,
        batch_size=args.batch_size or defaults.batch_size,
        num_workers=args.num_workers or defaults.num_workers,
    )
    structure_config = build_structure_config(args)
    if structure_config.use_ids:
        print("⚠️ IDS结构条件尚未接入数据源，将暂时忽略。")
        structure_config.use_ids = False

    base_dataset = PairedGlyphImageDataset(
        target_img_dir=dataset_config.target_img_dir,
        reference_img_dir=dataset_config.reference_img_dir,
        use_data_augmentation=False,
    )
    dataset = StructureConditionDataset(
        base_dataset=base_dataset,
        config=structure_config,
    )
    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
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

    vqgan2_config = VQGAN2ModelConfig()
    model_config = DiTModelConfig(token_dim=vqgan2_config.token_dim)
    training_config = DiTTrainingConfig(batch_size=dataset_config.batch_size)

    condition_channels = (
        int(structure_config.use_component_mask)
        + int(structure_config.use_edge_map)
        + int(structure_config.use_skeleton)
    )
    if condition_channels == 0:
        raise ValueError("结构条件通道为0，请启用至少一个结构条件。")

    model = DiTModel(
        token_dim=model_config.token_dim,
        time_steps=model_config.time_steps,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        dropout=model_config.dropout,
        window_size=model_config.window_size,
        shift_window=model_config.shift_window,
    ).to(device)
    tokenizer = VQGAN2Tokenizer(
        in_channels=vqgan2_config.input_img_channels,
        latent_dim=vqgan2_config.latent_dim,
        base_channels=vqgan2_config.base_channels,
        token_dim=vqgan2_config.token_dim,
        codebook_size=vqgan2_config.codebook_size,
        commitment_cost=vqgan2_config.commitment_cost,
        multiscale=vqgan2_config.multiscale,
        coarse_downsample=vqgan2_config.coarse_downsample,
        coarse_weight=vqgan2_config.coarse_weight,
        tanh_output=vqgan2_config.tanh_output,
        vq_decay=vqgan2_config.vq_decay,
        vq_epsilon=vqgan2_config.vq_epsilon,
    ).to(device)
    if args.vqgan_ckpt:
        checkpoint = torch.load(args.vqgan_ckpt, map_location=device)
        tokenizer.load_state_dict(checkpoint["tokenizer"])
        tokenizer.eval()
        for param in tokenizer.parameters():
            param.requires_grad = False
        print(f"✅ 已加载 VQGAN-2 tokenizer: {args.vqgan_ckpt}")
    downsample_factor = tokenizer.downsample_factor()
    condition_encoder = ConditionEncoder(
        in_channels=condition_channels,
        embed_dim=model_config.token_dim,
        downsample_factor=downsample_factor,
    ).to(device)
    print_model_params(model)

    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + list(condition_encoder.parameters())
        + (list(tokenizer.parameters()) if tokenizer.training else []),
        lr=training_config.learning_rate,
    )

    # 添加学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2000,
        T_mult=2,
        eta_min=1e-6
    )

    scheduler = DiffusionScheduler(steps=model_config.time_steps, device=device)

    model.train()
    condition_encoder.train()
    step = 0
    for epoch in range(training_config.num_epochs):
        for batch in train_loader:
            images = batch["image"].to(device)
            cond = batch["condition"].to(device)
            cond_tokens, cond_seq = condition_encoder(cond)
            tokens, vq_loss = tokenizer.encode(images)
            if not tokenizer.training:
                vq_loss = torch.zeros_like(vq_loss)
            if tokens.shape[-2] * tokens.shape[-1] > 128 * 128:
                raise ValueError(
                    "DiT token map is too large for Transformer attention. "
                    "Please regenerate dataset with smaller img_size "
                    "(e.g., 128x128) before training DiT."
                )

            timesteps = torch.randint(
                0, model_config.time_steps, (tokens.size(0),), device=device
            )
            noise = torch.randn_like(tokens)
            noisy_tokens = scheduler.add_noise(tokens, timesteps, noise)
            pred_noise = model(
                noisy_tokens,
                cond_tokens=cond_tokens,
                cond_seq=cond_seq,
                timesteps=timesteps,
            )
            loss = F.mse_loss(pred_noise, noise) + vq_loss
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(condition_encoder.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            lr_scheduler.step()

            if step % 10 == 0:
                print(
                    f"[DiT] step={step} loss={loss.item():.6f} vq={vq_loss.item():.6f}"
                )
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_path = Path(args.save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "dit": model.state_dict(),
                        "condition_encoder": condition_encoder.state_dict(),
                        "structure_config": structure_config.__dict__,
                    },
                    save_path,
                )
                print(f"✅ 已保存 DiT 模型: {save_path}")
                print("✅ 已达到最大训练步数，停止。")
                return


if __name__ == "__main__":
    main()
