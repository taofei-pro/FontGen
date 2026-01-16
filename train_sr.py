import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.sr_config import SRModelConfig, SRTrainingConfig
from datasets.image_dataset import GlyphImageDataset
from models.sr.sr_model import SRModel
from utils.hardware.hardware_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SR model")
    parser.add_argument("--img_dir", type=str, default="data/target")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="checkpoints/sr.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    defaults = SRModelConfig()
    model_config = SRModelConfig(
        scale=args.scale or defaults.scale,
        model_name=args.model_name or defaults.model_name,
    )
    training_config = SRTrainingConfig(
        batch_size=args.batch_size or SRTrainingConfig().batch_size
    )

    dataset = GlyphImageDataset(img_dir=args.img_dir, normalize=True)
    loader = DataLoader(
        dataset=dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=min(2, args.num_workers),
        pin_memory=True if device.type == "cuda" else False,
    )

    if model_config.model_name in {"realesrgan", "swinir"}:
        print(
            "Warning: external SR model training is not supported; "
            "falling back to EDSR."
        )
        model_config.model_name = "edsr"

    model = SRModel(scale=model_config.scale, model_name=model_config.model_name).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate
    )

    model.train()
    step = 0
    for epoch in range(training_config.num_epochs):
        for batch in loader:
            high_res = batch["tgt_img"].to(device)
            low_res = F.interpolate(
                high_res,
                scale_factor=1 / model_config.scale,
                mode="bilinear",
                align_corners=False,
            )
            pred = model(low_res)
            loss = F.mse_loss(pred, high_res)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"[SR] step={step} loss={loss.item():.6f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_path = Path(args.save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "sr": model.state_dict(),
                        "model_name": model_config.model_name,
                        "scale": model_config.scale,
                    },
                    save_path,
                )
                print(f"✅ 已保存 SR 模型: {save_path}")
                return


if __name__ == "__main__":
    main()
