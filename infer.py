import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from configs.dit_config import DiTModelConfig
from configs.structure_config import StructureConfig
from configs.vqgan_config import VQGAN2ModelConfig
from datasets.image_dataset import GlyphImageDataset
from datasets.structure_dataset import StructureConditionDataset
from models.dit.condition_encoder import ConditionEncoder
from models.dit.dit_model import DiTModel
from models.dit.scheduler import DiffusionScheduler
from models.sr.sr_model import SRModel
from models.sr.tiling import tile_infer
from models.vqgan.tokenizer import VQGAN2Tokenizer
from utils.hardware.hardware_utils import select_device
from utils.image.image_utils import convert_tensor_to_pil_images, save_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-gen inference pipeline")
    parser.add_argument("--condition_img_dir", type=str, default="data/target")
    parser.add_argument("--output_dir", type=str, default="data/outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vqgan_ckpt", type=str, default="checkpoints/vqgan.pth")
    parser.add_argument("--dit_ckpt", type=str, default="checkpoints/dit.pth")
    parser.add_argument("--sampling_steps", type=int, default=200)  # 增加采样步数，减少噪声残留
    parser.add_argument("--guidance_scale", type=float, default=7.0)  # 提高引导尺度，增强条件信息利用
    parser.add_argument("--cfg_rescale", type=float, default=0.7)  # 启用CFG重缩放，提高图像质量
    parser.add_argument("--x0_clip", type=float, default=1.0)  # 启用x0裁剪，稳定生成过程
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "karras"],
        default="karras",  # 使用Karras调度器，提高采样稳定性
    )
    parser.add_argument("--rho", type=float, default=7.0)  # Karras调度器参数
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["ddim", "ddpm", "dpmpp_2m", "dpmpp_2s", "dpmpp_3m"],
        default="dpmpp_2m",  # 使用DPM++ 2M采样器，平衡速度和质量
    )
    parser.add_argument("--use_component_mask", action="store_true")
    parser.add_argument("--use_edge_map", action="store_true")
    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_ids", action="store_true")
    parser.add_argument("--enable_sr", action="store_true")
    parser.add_argument("--sr_ckpt", type=str, default=None)
    parser.add_argument("--sr_model", type=str, default="basic")
    parser.add_argument("--sr_tile", action="store_true")
    parser.add_argument("--sr_tile_size", type=int, default=256)
    parser.add_argument("--num_chars", type=int, default=20, help="Number of characters to generate (default: 20)")
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


def load_structure_config_from_ckpt(
    ckpt_path: str,
    device: torch.device,
) -> StructureConfig | None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "structure_config" in checkpoint:
        return StructureConfig(**checkpoint["structure_config"])
    return None


@torch.no_grad()
def sample_tokens(
    model: DiTModel,
    condition_encoder: ConditionEncoder,
    scheduler: DiffusionScheduler,
    cond: torch.Tensor,
    steps: int,
    guidance_scale: float,
    sampler: str,
    schedule: str,
    rho: float,
    cfg_rescale: float,
    x0_clip: float | None,
) -> torch.Tensor:
    batch_size = cond.size(0)
    token_dim = model.token_dim
    cond_tokens, cond_seq = condition_encoder(cond)
    height, width = cond_tokens.shape[-2:]
    tokens = torch.randn(
        (batch_size, token_dim, height, width),
        device=cond.device,
        dtype=cond.dtype,
    )

    timesteps = scheduler.get_timesteps(steps, schedule=schedule, rho=rho)
    prev_preds: list[torch.Tensor] = []
    prev_lambdas: list[torch.Tensor] = []
    for i, t in enumerate(timesteps):
        t_int = int(t.item())
        t_prev = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else 0

        def predict_noise(x: torch.Tensor, t_val: int) -> torch.Tensor:
            t_tensor = torch.full(
                (batch_size,), t_val, device=cond.device, dtype=torch.long
            )
            if guidance_scale != 1.0:
                pred_uncond = model(x, cond_tokens=None, cond_seq=None, timesteps=t_tensor)
                pred_cond = model(x, cond_tokens=cond_tokens, cond_seq=cond_seq, timesteps=t_tensor)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                if cfg_rescale > 0:
                    std_pred = pred.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                    std_cond = pred_cond.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                    rescaled = pred * (std_cond / std_pred)
                    pred = pred * (1 - cfg_rescale) + rescaled * cfg_rescale
                return pred
            return model(x, cond_tokens=cond_tokens, cond_seq=cond_seq, timesteps=t_tensor)

        def maybe_clip_x0(pred_x0: torch.Tensor) -> torch.Tensor:
            if x0_clip is None:
                return pred_x0
            return pred_x0.clamp(-x0_clip, x0_clip)

        def predict_x0(x: torch.Tensor, noise: torch.Tensor, t_val: int) -> torch.Tensor:
            alpha_t, sigma_t, _ = scheduler.get_alpha_sigma_lambda(t_val)
            pred_x0 = (x - sigma_t * noise) / alpha_t
            return maybe_clip_x0(pred_x0)

        pred_noise = predict_noise(tokens, t_int)
        if sampler == "ddpm":
            tokens = scheduler.ddpm_step(tokens, t_int, pred_noise)
        elif sampler == "dpmpp_2m":
            alpha_t, sigma_t, lambda_t = scheduler.get_alpha_sigma_lambda(t_int)
            alpha_s, sigma_s, lambda_s = scheduler.get_alpha_sigma_lambda(t_prev)
            pred_x0 = (tokens - sigma_t * pred_noise) / alpha_t
            pred_x0 = maybe_clip_x0(pred_x0)
            if t_prev == 0:
                tokens = pred_x0
                prev_preds.append(pred_noise)
                prev_lambdas.append(lambda_t)
                continue
            if len(prev_preds) < 1 or len(prev_lambdas) < 1:
                pred_noise_hat = pred_noise
            else:
                h = lambda_s - lambda_t
                h_prev = lambda_t - prev_lambdas[-1]
                r = h_prev / h
                pred_noise_hat = (1 + 1 / (2 * r)) * pred_noise - (
                    1 / (2 * r)
                ) * prev_preds[-1]
            tokens = alpha_s * pred_x0 + sigma_s * pred_noise_hat
            prev_preds.append(pred_noise)
            prev_lambdas.append(lambda_t)
        elif sampler == "dpmpp_2s":
            alpha_t, sigma_t, lambda_t = scheduler.get_alpha_sigma_lambda(t_int)
            alpha_s, sigma_s, lambda_s = scheduler.get_alpha_sigma_lambda(t_prev)
            pred_x0 = (tokens - sigma_t * pred_noise) / alpha_t
            pred_x0 = maybe_clip_x0(pred_x0)
            if t_prev == 0:
                tokens = pred_x0
                prev_preds.append(pred_noise)
                prev_lambdas.append(lambda_t)
                continue
            lambda_mid = lambda_t + 0.5 * (lambda_s - lambda_t)
            alpha_m, sigma_m = scheduler.alpha_sigma_from_lambda(lambda_mid)
            t_mid = int(scheduler.t_from_lambda(lambda_mid.unsqueeze(0))[0].item())
            x_mid = alpha_m * pred_x0 + sigma_m * pred_noise
            pred_noise_mid = predict_noise(x_mid, t_mid)
            tokens = alpha_s * pred_x0 + sigma_s * pred_noise_mid
            prev_preds.append(pred_noise)
            prev_lambdas.append(lambda_t)
        elif sampler == "dpmpp_3m":
            alpha_t, sigma_t, lambda_t = scheduler.get_alpha_sigma_lambda(t_int)
            alpha_s, sigma_s, lambda_s = scheduler.get_alpha_sigma_lambda(t_prev)
            pred_x0 = (tokens - sigma_t * pred_noise) / alpha_t
            pred_x0 = maybe_clip_x0(pred_x0)
            if t_prev == 0:
                tokens = pred_x0
                prev_preds.append(pred_noise)
                prev_lambdas.append(lambda_t)
                continue
            if len(prev_preds) < 2 or len(prev_lambdas) < 2:
                pred_noise_hat = pred_noise
            else:
                lam0, lam1, lam2 = lambda_t, prev_lambdas[-1], prev_lambdas[-2]
                eps0, eps1, eps2 = pred_noise, prev_preds[-1], prev_preds[-2]
                l0 = (lambda_s - lam1) * (lambda_s - lam2) / (
                    (lam0 - lam1) * (lam0 - lam2)
                )
                l1 = (lambda_s - lam0) * (lambda_s - lam2) / (
                    (lam1 - lam0) * (lam1 - lam2)
                )
                l2 = (lambda_s - lam0) * (lambda_s - lam1) / (
                    (lam2 - lam0) * (lam2 - lam1)
                )
                pred_noise_hat = l0 * eps0 + l1 * eps1 + l2 * eps2
            tokens = alpha_s * pred_x0 + sigma_s * pred_noise_hat
            prev_preds.append(pred_noise)
            prev_lambdas.append(lambda_t)
        else:
            if t_prev == 0:
                tokens = predict_x0(tokens, pred_noise, t_int)
            else:
                tokens = scheduler.ddim_step(tokens, t_int, pred_noise)

        # Keep last 2 history entries to stabilize 2M/3M
        if len(prev_preds) > 2:
            prev_preds = prev_preds[-2:]
        if len(prev_lambdas) > 2:
            prev_lambdas = prev_lambdas[-2:]
    return tokens


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    structure_config = load_structure_config_from_ckpt(args.dit_ckpt, device)
    if structure_config is None:
        structure_config = build_structure_config(args)
    if structure_config.use_ids:
        print("⚠️ IDS结构条件尚未接入数据源，将暂时忽略。")
        structure_config.use_ids = False

    dataset = GlyphImageDataset(img_dir=args.condition_img_dir, normalize=True)
    dataset = StructureConditionDataset(base_dataset=dataset, config=structure_config)
    
    # Limit to num_chars if specified
    if args.num_chars > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.num_chars, len(dataset))))
        print(f"[Infer] Limited to {len(dataset)} characters")
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=True if device.type == "cuda" else False,
    )

    vqgan2_config = VQGAN2ModelConfig()
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
    dit_config = DiTModelConfig(token_dim=vqgan2_config.token_dim)
    model = DiTModel(
        token_dim=dit_config.token_dim,
        time_steps=dit_config.time_steps,
        num_layers=dit_config.num_layers,
        num_heads=dit_config.num_heads,
        mlp_ratio=dit_config.mlp_ratio,
        dropout=dit_config.dropout,
        window_size=dit_config.window_size,
        shift_window=dit_config.shift_window,
    ).to(device)
    condition_channels = (
        int(structure_config.use_component_mask)
        + int(structure_config.use_edge_map)
        + int(structure_config.use_skeleton)
    )

    vq_ckpt = torch.load(args.vqgan_ckpt, map_location=device)
    tokenizer.load_state_dict(vq_ckpt["tokenizer"])
    tokenizer.eval()

    downsample_factor = tokenizer.downsample_factor()
    condition_encoder = ConditionEncoder(
        in_channels=condition_channels,
        embed_dim=dit_config.token_dim,
        downsample_factor=downsample_factor,
    ).to(device)

    dit_ckpt = torch.load(args.dit_ckpt, map_location=device)
    model.load_state_dict(dit_ckpt["dit"])
    condition_encoder.load_state_dict(dit_ckpt["condition_encoder"])
    model.eval()
    condition_encoder.eval()

    sr_model = None
    if args.enable_sr:
        sr_model = SRModel(
            scale=2, model_name=args.sr_model, ckpt_path=args.sr_ckpt
        ).to(device)
        if args.sr_ckpt:
            sr_ckpt = torch.load(args.sr_ckpt, map_location=device)
            if sr_model.is_torchscript:
                sr_ckpt = None
            elif "model_name" in sr_ckpt:
                sr_model = SRModel(
                    scale=sr_ckpt.get("scale", 2),
                    model_name=sr_ckpt["model_name"],
                ).to(device)
                sr_model.load_state_dict(sr_ckpt["sr"])
            else:
                sr_model.load_state_dict(sr_ckpt["sr"])
        sr_model.eval()

    scheduler = DiffusionScheduler(steps=dit_config.time_steps, device=device)
    output_dir = Path(args.output_dir)

    for batch in tqdm(loader, desc="Sampling", unit="batch"):
        cond = batch["condition"].to(device)
        tokens = sample_tokens(
            model=model,
            condition_encoder=condition_encoder,
            scheduler=scheduler,
            cond=cond,
            steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            sampler=args.sampler,
            schedule=args.schedule,
            rho=args.rho,
            cfg_rescale=args.cfg_rescale,
            x0_clip=args.x0_clip,
        )
        
        images = tokenizer.decode(tokens)

        # 完全移除后处理，只保留最基本的clamp操作
        # 以获得最原始的生成结果
        images = images.clamp(-1.0, 1.0)
        
        # 添加降噪处理，减少生成图像中的噪点
        # 1. 自定义高斯模糊降噪（减少随机噪点）
        def gaussian_blur(image, kernel_size=3, sigma=0.8):
            # 创建高斯核
            k = kernel_size
            t = torch.linspace(-(k//2), k//2, k)
            gauss = torch.exp(-t**2 / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            # 2D高斯核
            gauss_kernel = gauss.view(-1, 1) * gauss.view(1, -1)
            gauss_kernel = gauss_kernel.view(1, 1, k, k)
            # 应用卷积
            padding = k // 2
            blurred = torch.nn.functional.conv2d(image, gauss_kernel.to(image.device), padding=padding, groups=image.size(1))
            return blurred
        
        images = gaussian_blur(images, kernel_size=3, sigma=0.8)
        # 2. 自适应阈值处理（增强轮廓，减少噪声）
        images_01 = (images + 1.0) / 2.0  # 转换到[0,1]范围
        # 计算局部均值作为阈值
        local_mean = torch.nn.functional.avg_pool2d(images_01, kernel_size=5, stride=1, padding=2)
        # 应用自适应阈值
        images_01 = torch.where(images_01 > local_mean - 0.1, images_01, torch.zeros_like(images_01))
        # 转换回[-1,1]范围
        images = images_01 * 2.0 - 1.0

        if images.numel() > 0:
            img_min = images.min().item()
            img_max = images.max().item()
            img_mean = images.mean().item()
            print(f"[Infer] pre-SR stats min={img_min:.3f} max={img_max:.3f} mean={img_mean:.3f}")

        if sr_model is not None:
            if args.sr_tile:
                images = tile_infer(sr_model, images, tile_size=args.sr_tile_size)
            else:
                images = sr_model(images)
            images = images.clamp(-1.0, 1.0)
            if images.numel() > 0:
                img_min = images.min().item()
                img_max = images.max().item()
                img_mean = images.mean().item()
                print(f"[Infer] post-SR stats min={img_min:.3f} max={img_max:.3f} mean={img_mean:.3f}")

        pil_images = convert_tensor_to_pil_images(images)
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        img_names = []
        if isinstance(batch.get("meta"), dict) and "img_name" in batch["meta"]:
            img_names = list(batch["meta"]["img_name"])
        if not img_names:
            img_names = [f"sample_{i}" for i in range(len(pil_images))]
        save_images(pil_images, img_names, output_dir)


if __name__ == "__main__":
    main()
