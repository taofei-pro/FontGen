import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs import VQGAN2DatasetConfig
from datasets.dataset_utils import save_dataset_charset, split_dataset
from datasets.image_dataset import PairedGlyphImageDataset
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract train/val charset from dataset"
    )
    parser.add_argument("--target_font_path", type=str, help="Target font path")
    parser.add_argument("--split_ratios", type=float, nargs=2, help="Train/val ratios")
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, help="Training device (mps, cpu, cuda)")
    return parser.parse_args()


def extract_train_val_charset(
    target_font_path: str,
    dataset_config: VQGAN2DatasetConfig,
    device: torch.device,
) -> None:
    dataset = PairedGlyphImageDataset(
        target_img_dir=dataset_config.target_img_dir,
        reference_img_dir=dataset_config.reference_img_dir,
        use_data_augmentation=False,
    )
    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        split_ratios=dataset_config.split_ratios,
        random_seed=dataset_config.random_seed,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=min(2, dataset_config.num_workers),
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=1 if dataset_config.num_workers > 0 else None,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=min(2, dataset_config.num_workers),
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=1 if dataset_config.num_workers > 0 else None,
        persistent_workers=False,
    )
    save_dataset_charset(
        train_loader=train_loader,
        val_loader=val_loader,
        target_font_path=target_font_path,
        charset_root="charsets",
    )
    train_count = len(train_dataset)
    val_count = len(val_dataset)
    target_font_name = Path(target_font_path).stem
    print(f"✅ 已生成字集拆分：{target_font_name}")
    print(
        f"   - train: {train_count} samples -> charsets/splits/{target_font_name}/train.txt"
    )
    print(
        f"   - val:   {val_count} samples -> charsets/splits/{target_font_name}/val.txt"
    )


def main() -> None:
    args = parse_args()
    dataset_config = update_config_from_args(
        converting_config=VQGAN2DatasetConfig(),
        args=args,
    )
    device = select_device(args.device)
    extract_train_val_charset(
        target_font_path=args.target_font_path,
        dataset_config=dataset_config,
        device=device,
    )


if __name__ == "__main__":
    main()
