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
from utils.image.image_generator import GlyphImageGenerator
from configs.font_processing_config import FontProcessingConfig
from utils.font.font_utils import read_charset_from_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default='data/reference', help='Reference images directory')
    parser.add_argument('--output_dir', type=str, default='data/outputs', help='Output directory')
    parser.add_argument('--vqvae_checkpoint', type=str, default='checkpoints/vqvae.pth', help='VQ-VAE checkpoint path')
    parser.add_argument('--ldm_checkpoint', type=str, default='checkpoints/ldm.pth', help='LDM checkpoint path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--sample_steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--charset_path', type=str, default='charsets/target_charset.txt', help='Charset file path')
    parser.add_argument('--reference_fonts_dir', type=str, default='fonts/jigmo', help='Reference fonts directory')
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

def generate_images_from_charset(model, charset_path, reference_fonts_dir, output_dir, batch_size, sample_steps, device):
    """从字符集文件生成字形"""
    # 加载字符集
    charset = read_charset_from_file(charset_path)
    
    # 配置字体处理
    font_processing_config = FontProcessingConfig()
    
    # 使用data/reference目录
    ref_dir = Path('data/reference')
    ref_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成reference图片
    ref_generators = GlyphImageGenerator.from_reference_fonts(
        reference_fonts_dir=reference_fonts_dir,
        font_processing_config=font_processing_config,
    )
    
    # 为每个字符生成reference图片
    for char in tqdm(charset, desc='Generating reference images'):
        for ref_generator in ref_generators:
            # 检查字符是否在字体的覆盖范围内
            covered_charset_path = (
                Path(font_processing_config.unihan_coverage_charset_dir)
                / ref_generator.font_name
                / 'covered.txt'
            )
            if covered_charset_path.exists():
                covered_charset = read_charset_from_file(covered_charset_path)
                if char in covered_charset:
                    ref_generator.save_glyph_image(
                        char=char,
                        output_dir=ref_dir,
                        img_size=font_processing_config.img_size,
                    )
                    break  # 只使用第一个能覆盖该字符的字体
    
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
    
    # VQ-VAE configuration
    vqvae_config = {
        'in_channels': 1,
        'base_channels': 64,
        'latent_dim': 2,
        'codebook_size': 64,
        'commitment_cost': 0.25
    }
    
    # LDM configuration
    ldm_config = {
        'time_pos_dim': 64,
        'time_emb_dim': 256,
        'time_steps': 1000,
        'unet_base_channels': 64
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
        reference_fonts_dir=args.reference_fonts_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        device=device
    )

if __name__ == '__main__':
    main()