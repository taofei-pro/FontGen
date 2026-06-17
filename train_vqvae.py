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

from models.vqvae.vqvae import VQVAE
from datasets.image_dataset import PairedGlyphImageDataset
from configs.vqvae_config import VQVAEModelConfig, VQVAETrainingConfig, VQVAEDatasetConfig

def parse_args():
    # Load default configs
    dataset_config = VQVAEDatasetConfig()
    training_config = VQVAETrainingConfig()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_dir', type=str, default=dataset_config.target_img_dir, help='Target images directory')
    parser.add_argument('--ref_dir', type=str, default=dataset_config.reference_img_dir, help='Reference images directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default=training_config.save_dir, help='Checkpoint save directory')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Sample images save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--use_amp', type=bool, default=True, help='Use automatic mixed precision')
    parser.add_argument('--split_ratios', type=float, nargs=2, default=list(dataset_config.split_ratios), help='Train/val split ratios')
    parser.add_argument('--random_seed', type=int, default=dataset_config.random_seed, help='Random seed for data splitting')
    parser.add_argument('--img_save_interval', type=int, default=5, help='Image save interval (epochs)')
    return parser.parse_args()

def save_samples(model, dataloader, sample_dir, epoch, device):
    """Save sample images for visualization and compute evaluation metrics"""
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
        
        tgt_recon, _ = model(tgt_imgs)
        ref_recon, _ = model(ref_imgs)
        
        # Compute metrics for target images
        tgt_ssim = ssim(tgt_recon, tgt_imgs)
        tgt_psnr = psnr(tgt_recon, tgt_imgs)
        total_ssim += tgt_ssim.item()
        total_psnr += tgt_psnr.item()
        num_samples += 1
        
        # Compute metrics for reference images
        ref_ssim = ssim(ref_recon, ref_imgs)
        ref_psnr = psnr(ref_recon, ref_imgs)
        total_ssim += ref_ssim.item()
        total_psnr += ref_psnr.item()
        num_samples += 1
        
        # Save samples
        for i, (tgt, tgt_r, ref, ref_r) in enumerate(zip(tgt_imgs, tgt_recon, ref_imgs, ref_recon)):
            # Convert tensors to PIL images
            def tensor_to_pil(tensor):
                img = tensor.squeeze().cpu().numpy()
                img = (img + 1) / 2 * 255  # Normalize from [-1, 1] to [0, 255]
                img = img.astype(np.uint8)
                return Image.fromarray(img, mode='L')
            
            tgt_pil = tensor_to_pil(tgt)
            tgt_r_pil = tensor_to_pil(tgt_r)
            ref_pil = tensor_to_pil(ref)
            ref_r_pil = tensor_to_pil(ref_r)
            
            # Create combined image
            width, height = tgt_pil.size
            combined = Image.new('L', (width * 4, height))
            combined.paste(tgt_pil, (0, 0))
            combined.paste(tgt_r_pil, (width, 0))
            combined.paste(ref_pil, (width * 2, 0))
            combined.paste(ref_r_pil, (width * 3, 0))
            
            combined.save(sample_dir / f'sample_{i}.png')
    
    # Calculate average metrics
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    
    print(f'Saved samples to {sample_dir}')
    print(f'Reconstruction metrics - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}')

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    sample_dir = Path(args.sample_dir) / 'vqvae'
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize dataset and split into train/val
    dataset = PairedGlyphImageDataset(
        args.tgt_dir, 
        args.ref_dir,
        use_data_augmentation=True,
        augmentation_type="advanced",
    )
    train_size = int(len(dataset) * args.split_ratios[0])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.random_seed)
    )
    
    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize VQ-VAE model using config
    model_config = VQVAEModelConfig()
    model = VQVAE(
        in_channels=model_config.in_channels,
        base_channels=model_config.base_channels,
        latent_dim=model_config.latent_dim,
        codebook_size=model_config.codebook_size,
        commitment_cost=model_config.commitment_cost
    ).to(device)
    
    # Initialize optimizer, scheduler, and scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_total_recon_loss = 0.0
        train_total_vq_loss = 0.0
        
        with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for batch in pbar:
                tgt_imgs = batch['tgt_img'].to(device)
                ref_imgs = batch['ref_img'].to(device)
                
                # Forward pass
                with autocast(device_type=device.type, enabled=args.use_amp):
                    tgt_recon, tgt_vq_loss = model(tgt_imgs)
                    tgt_recon_loss = torch.nn.functional.mse_loss(tgt_recon, tgt_imgs)
                    
                    ref_recon, ref_vq_loss = model(ref_imgs)
                    ref_recon_loss = torch.nn.functional.mse_loss(ref_recon, ref_imgs)
                    
                    recon_loss = (tgt_recon_loss + ref_recon_loss) / 2
                    vq_loss = (tgt_vq_loss + ref_vq_loss) / 2
                    loss = recon_loss + vq_loss
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                train_total_loss += loss.item()
                train_total_recon_loss += recon_loss.item()
                train_total_vq_loss += vq_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'vq_loss': vq_loss.item()
                })
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_total_recon_loss = 0.0
        val_total_vq_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{args.num_epochs}') as pbar:
                for batch in pbar:
                    tgt_imgs = batch['tgt_img'].to(device)
                    ref_imgs = batch['ref_img'].to(device)
                    
                    with autocast(device_type=device.type, enabled=args.use_amp):
                        tgt_recon, tgt_vq_loss = model(tgt_imgs)
                        tgt_recon_loss = torch.nn.functional.mse_loss(tgt_recon, tgt_imgs)
                        
                        ref_recon, ref_vq_loss = model(ref_imgs)
                        ref_recon_loss = torch.nn.functional.mse_loss(ref_recon, ref_imgs)
                        
                        recon_loss = (tgt_recon_loss + ref_recon_loss) / 2
                        vq_loss = (tgt_vq_loss + ref_vq_loss) / 2
                        loss = recon_loss + vq_loss
                    
                    val_total_loss += loss.item()
                    val_total_recon_loss += recon_loss.item()
                    val_total_vq_loss += vq_loss.item()
                    
                    pbar.set_postfix({
                        'val_loss': loss.item(),
                        'val_recon_loss': recon_loss.item(),
                        'val_vq_loss': vq_loss.item()
                    })
        
        # Calculate average losses
        train_avg_loss = train_total_loss / len(train_loader)
        train_avg_recon_loss = train_total_recon_loss / len(train_loader)
        train_avg_vq_loss = train_total_vq_loss / len(train_loader)
        
        val_avg_loss = val_total_loss / len(val_loader)
        val_avg_recon_loss = val_total_recon_loss / len(val_loader)
        val_avg_vq_loss = val_total_vq_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print metrics
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        print(f'Training - Loss: {train_avg_loss:.6f}, Recon Loss: {train_avg_recon_loss:.6f}, VQ Loss: {train_avg_vq_loss:.6f}')
        print(f'Validation - Loss: {val_avg_loss:.6f}, Recon Loss: {val_avg_recon_loss:.6f}, VQ Loss: {val_avg_vq_loss:.6f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save samples
        if (epoch + 1) % args.img_save_interval == 0 or epoch == args.num_epochs - 1:
            save_samples(model, val_loader, sample_dir, epoch + 1, device)
        
        # Save best model
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            checkpoint_path = save_dir / 'vqvae.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')

if __name__ == '__main__':
    main()