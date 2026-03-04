import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from models.ldm.ldm import LDM
from datasets.image_dataset import PairedGlyphImageDataset
from configs.ldm_config import LDMModelConfig, LDMTrainingConfig
from configs.vqvae_config import VQVAEModelConfig, VQVAEDatasetConfig

def parse_args():
    # Load default configs
    dataset_config = VQVAEDatasetConfig()
    training_config = LDMTrainingConfig()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_dir', type=str, default=dataset_config.target_img_dir, help='Target images directory')
    parser.add_argument('--ref_dir', type=str, default=dataset_config.reference_img_dir, help='Reference images directory')
    parser.add_argument('--vqvae_checkpoint', type=str, default=training_config.vqvae_checkpoint, help='VQ-VAE checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default=training_config.save_dir, help='Checkpoint save directory')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Sample images save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--use_amp', type=bool, default=True, help='Use automatic mixed precision')
    parser.add_argument('--split_ratios', type=float, nargs=2, default=list(dataset_config.split_ratios), help='Train/val split ratios')
    parser.add_argument('--random_seed', type=int, default=dataset_config.random_seed, help='Random seed for data splitting')
    parser.add_argument('--img_save_interval', type=int, default=5, help='Image save interval (epochs)')
    parser.add_argument('--sample_steps', type=int, default=50, help='Number of sampling steps')
    return parser.parse_args()

def save_samples(model, dataloader, sample_dir, epoch, device, sample_steps):
    """Save generated samples for visualization and compute evaluation metrics"""
    model.eval()
    sample_dir = Path(sample_dir) / f'epoch_{epoch:04d}'
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)  # [-1, 1] range
    psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)  # [-1, 1] range
    
    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        tgt_imgs = batch['tgt_img'].to(device)
        ref_imgs = batch['ref_img'].to(device)
        
        # Generate images
        generated_imgs = model.generate(ref_imgs, sample_steps=sample_steps)
        
        # Compute metrics
        current_ssim = ssim(generated_imgs, tgt_imgs)
        current_psnr = psnr(generated_imgs, tgt_imgs)
        total_ssim += current_ssim.item()
        total_psnr += current_psnr.item()
        num_samples += 1
        
        # Save samples
        for i, (tgt, ref, gen) in enumerate(zip(tgt_imgs, ref_imgs, generated_imgs)):
            # Convert tensors to PIL images
            def tensor_to_pil(tensor):
                img = tensor.squeeze().cpu().numpy()
                img = (img + 1) / 2 * 255  # Normalize from [-1, 1] to [0, 255]
                img = img.astype(np.uint8)
                return Image.fromarray(img, mode='L')
            
            tgt_pil = tensor_to_pil(tgt)
            ref_pil = tensor_to_pil(ref)
            gen_pil = tensor_to_pil(gen)
            
            # Create combined image
            width, height = tgt_pil.size
            combined = Image.new('L', (width * 3, height))
            combined.paste(tgt_pil, (0, 0))
            combined.paste(ref_pil, (width, 0))
            combined.paste(gen_pil, (width * 2, 0))
            
            combined.save(sample_dir / f'sample_{i}.png')
    
    # Calculate average metrics
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    
    print(f'Saved samples to {sample_dir}')
    print(f'Generation metrics - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}')

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    sample_dir = Path(args.sample_dir) / 'ldm'
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize dataset and split into train/val
    dataset = PairedGlyphImageDataset(args.tgt_dir, args.ref_dir)
    train_size = int(len(dataset) * args.split_ratios[0])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.random_seed)
    )
    
    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    
    # Load VQ-VAE checkpoint and freeze it
    model.load_vqvae_checkpoint(args.vqvae_checkpoint)
    model.freeze_vqvae()
    
    # Initialize optimizer, scheduler, and scaler
    optimizer = optim.Adam(model.unet.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0.0
        
        with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for batch in pbar:
                # Forward pass
                with autocast(device_type=device.type, enabled=args.use_amp):
                    loss = model.train_step(batch)
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                train_total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{args.num_epochs}') as pbar:
                for batch in pbar:
                    with autocast(device_type=device.type, enabled=args.use_amp):
                        loss = model.train_step(batch)
                    
                    val_total_loss += loss.item()
                    
                    pbar.set_postfix({'val_loss': loss.item()})
        
        # Calculate average losses
        train_avg_loss = train_total_loss / len(train_loader)
        val_avg_loss = val_total_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print metrics
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        print(f'Training - Loss: {train_avg_loss:.6f}')
        print(f'Validation - Loss: {val_avg_loss:.6f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save samples
        if (epoch + 1) % args.img_save_interval == 0 or epoch == args.num_epochs - 1:
            save_samples(model, val_loader, sample_dir, epoch + 1, device, args.sample_steps)
        
        # Save best model
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            checkpoint_path = save_dir / 'ldm.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = save_dir / f'ldm_epoch_{epoch+1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_avg_loss
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

if __name__ == '__main__':
    main()