import torch
import torchvision.transforms as T
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm

from models.ldm.ldm import LDM
from utils.font.font_utils import read_charset_from_file
from utils.image.image_utils import get_image_paths
from configs.ldm_config import LDMModelConfig
from configs.vqvae_config import VQVAEModelConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default='data/reference_infer', help='Reference images directory')
    parser.add_argument('--output_dir', type=str, default='data/outputs', help='Output directory')
    parser.add_argument('--vqvae_checkpoint', type=str, default='checkpoints/vqvae.pth', help='VQ-VAE checkpoint path')
    parser.add_argument('--ldm_checkpoint', type=str, default='checkpoints/ldm.pth', help='LDM checkpoint path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--sample_steps', type=int, default=250, help='Number of sampling steps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM sampling eta')
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
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def generate_images_from_charset(model, charset_path, ref_dir, output_dir, batch_size, sample_steps, device, eta):
    """从字符集文件生成字形"""
    # 加载字符集
    charset = read_charset_from_file(charset_path)
    
    # 加载参考图像
    ref_img_paths = get_image_paths(ref_dir)
    ref_img_dict = {Path(path).stem: path for path in ref_img_paths}
    
    # 创建变换
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    # 生成图像
    with torch.no_grad():
        for char in tqdm(charset, desc='Generating images'):
            # 获取字符的编码
            char_code = f"{ord(char):05X}"
            
            # 检查是否有对应的参考图像
            if char_code in ref_img_dict:
                # 加载参考图像
                ref_img_path = ref_img_dict[char_code]
                ref_img = Image.open(ref_img_path).convert("L")
                ref_img = transform(ref_img).unsqueeze(0).to(device)
                
                # 生成图像
                generated_imgs = model.generate(ref_img, sample_steps=sample_steps, eta=eta)
                
                # 保存生成的图像
                output_path = output_dir / f"{char_code}.png"
                save_image(generated_imgs[0], output_path)
                print(f'Saved generated image to {output_path}')
            else:
                pass

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize LDM model using config
    ldm_model_config = LDMModelConfig()
    vqvae_model_config = VQVAEModelConfig()
    
    # Convert config to the format expected by LDM
    vqvae_config = {
        'in_channels': vqvae_model_config.in_channels,
        'base_channels': vqvae_model_config.base_channels,
        'latent_dim': vqvae_model_config.latent_dim,
        'codebook_size': vqvae_model_config.codebook_size,
        'commitment_cost': vqvae_model_config.commitment_cost
    }
    
    ldm_config = {
        'time_pos_dim': ldm_model_config.time_pos_dim,
        'time_emb_dim': ldm_model_config.time_emb_dim,
        'time_steps': ldm_model_config.time_steps,
        'unet_base_channels': ldm_model_config.unet_base_channels
    }
    
    # Initialize LDM model
    model = LDM(vqvae_config, ldm_config, device).to(device)
    
    # Load checkpoints
    model.load_vqvae_checkpoint(args.vqvae_checkpoint)
    ldm_checkpoint = torch.load(args.ldm_checkpoint, map_location=device)
    # Check if checkpoint contains model_state_dict
    if 'model_state_dict' in ldm_checkpoint:
        model.load_state_dict(ldm_checkpoint['model_state_dict'])
    else:
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
        device=device,
        eta=args.eta
    )

if __name__ == '__main__':
    main()