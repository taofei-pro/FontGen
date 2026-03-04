import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from models.ldm.ldm import LDM
from datasets.image_dataset import PairedGlyphImageDataset
from utils.font.font_utils import read_charset_from_file
from configs.ldm_config import LDMModelConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default='data/reference_infer', help='Reference images directory')
    parser.add_argument('--output_dir', type=str, default='data/outputs', help='Output directory')
    parser.add_argument('--vqvae_checkpoint', type=str, default='checkpoints/vqvae.pth', help='VQ-VAE checkpoint path')
    parser.add_argument('--ldm_checkpoint', type=str, default='checkpoints/ldm.pth', help='LDM checkpoint path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--sample_steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--charset_path', type=str, default='charsets/target_charset.txt', help='Charset file path')
    return parser.parse_args()

def save_image(tensor, path):
    # Convert tensor to PIL image
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).to(torch.uint8)
    img = Image.fromarray(tensor.cpu().numpy(), mode='L')
    # Ensure path has .png extension
    if not str(path).endswith('.png'):
        path = Path(str(path) + '.png')
    img.save(path)

def generate_images_from_charset(model, charset_path, ref_dir, output_dir, batch_size, sample_steps, device):
    """从字符集文件生成字形"""
    # 加载字符集
    charset = read_charset_from_file(charset_path)
    
    # 创建数据集和数据加载器
    dataset = PairedGlyphImageDataset(ref_dir, ref_dir)  # 使用相同的目录作为target和reference
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 生成图像
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating images'):
            ref_imgs = batch['ref_img'].to(device)
            img_names = batch['img_name']
            
            # 生成图像
            generated_imgs = model.generate(ref_imgs, sample_steps=sample_steps)
            
            # 保存生成的图像
            for i, img_name in enumerate(img_names):
                output_path = output_dir / img_name
                save_image(generated_imgs[i], output_path)
                print(f'Saved generated image to {output_path}')

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize LDM model using config
    model_config = LDMModelConfig()
    
    # Convert config to the format expected by LDM
    vqvae_config = {
        'in_channels': model_config.vqvae_in_channels,
        'base_channels': model_config.vqvae_base_channels,
        'latent_dim': model_config.vqvae_latent_dim,
        'codebook_size': model_config.vqvae_codebook_size,
        'commitment_cost': model_config.vqvae_commitment_cost
    }
    
    ldm_config = {
        'time_pos_dim': model_config.time_pos_dim,
        'time_emb_dim': model_config.time_emb_dim,
        'time_steps': model_config.time_steps,
        'unet_base_channels': model_config.unet_base_channels
    }
    
    # Initialize LDM model
    model = LDM(vqvae_config, ldm_config, device).to(device)
    
    # Load checkpoints
    model.load_vqvae_checkpoint(args.vqvae_checkpoint)
    ldm_checkpoint = torch.load(args.ldm_checkpoint, map_location=device)
    model.load_state_dict(ldm_checkpoint)
    model.eval()
    
    # 从字符集生成图像
    generate_images_from_charset(
        model=model,
        charset_path=args.charset_path,
        ref_dir=Path(args.ref_dir),
        output_dir=output_dir,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        device=device
    )

if __name__ == '__main__':
    main()