import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from utils.hardware.hardware_utils import select_device
from utils.image.image_utils import check_names_match, get_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute similarity metrics")
    parser.add_argument("--gen_dir", type=str, default="data/outputs")
    parser.add_argument("--gt_dir", type=str, default="data/target")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resize_gen_to_gt", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def compute_metrics(
    gen_dir: Path,
    gt_dir: Path,
    batch_size: int,
    device: torch.device,
    resize_gen_to_gt: bool,
    skip_lpips: bool,
) -> None:
    gen_paths = get_image_paths(gen_dir)
    gt_paths = get_image_paths(gt_dir)
    
    # Create a dictionary of ground truth paths by filename
    gt_dict = {path.name: path for path in gt_paths}
    
    # Filter generated paths to only include those with matching ground truth
    filtered_gen_paths = []
    filtered_gt_paths = []
    for gen_path in gen_paths:
        if gen_path.name in gt_dict:
            filtered_gen_paths.append(gen_path)
            filtered_gt_paths.append(gt_dict[gen_path.name])
    
    if not filtered_gen_paths:
        raise ValueError("No matching files found between gen_dir and gt_dir.")
    
    print(f"Found {len(filtered_gen_paths)} matching files out of {len(gen_paths)} generated files.")
    
    # Update paths to use filtered lists
    gen_paths = filtered_gen_paths
    gt_paths = filtered_gt_paths

    lpips_model = None
    if not skip_lpips:
        import lpips  # type: ignore

        lpips_model = lpips.LPIPS(net="vgg").eval().to(device)

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    for i in tqdm(range(0, len(gen_paths), batch_size), desc="Computing metrics"):
        batch_gen = gen_paths[i : i + batch_size]
        batch_gt = gt_paths[i : i + batch_size]
        for gen_path, gt_path in zip(batch_gen, batch_gt):
            gen_img = load_image(gen_path)
            gt_img = load_image(gt_path)
            if resize_gen_to_gt and gen_img.size != gt_img.size:
                gen_img = gen_img.resize(gt_img.size, Image.BICUBIC)

            gen_np = np.asarray(gen_img, dtype=np.float32) / 255.0
            gt_np = np.asarray(gt_img, dtype=np.float32) / 255.0

            psnr_scores.append(
                peak_signal_noise_ratio(gt_np, gen_np, data_range=1.0)
            )
            ssim_scores.append(
                structural_similarity(gt_np, gen_np, data_range=1.0)
            )

            if lpips_model is not None:
                gen_t = torch.from_numpy(gen_np).unsqueeze(0).unsqueeze(0)
                gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)
                gen_t = gen_t.repeat(1, 3, 1, 1) * 2 - 1
                gt_t = gt_t.repeat(1, 3, 1, 1) * 2 - 1
                gen_t = gen_t.to(device)
                gt_t = gt_t.to(device)
                lpips_scores.append(lpips_model(gen_t, gt_t).mean().item())

    def avg(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    print("Metrics:")
    print(f"  PSNR:  {avg(psnr_scores):.4f}")
    print(f"  SSIM:  {avg(ssim_scores):.4f}")
    if lpips_model is not None:
        print(f"  LPIPS: {avg(lpips_scores):.4f}")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    compute_metrics(
        gen_dir=Path(args.gen_dir),
        gt_dir=Path(args.gt_dir),
        batch_size=args.batch_size,
        device=device,
        resize_gen_to_gt=args.resize_gen_to_gt,
        skip_lpips=args.skip_lpips,
    )


if __name__ == "__main__":
    main()
