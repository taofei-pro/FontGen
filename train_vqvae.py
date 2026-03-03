import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
from pathlib import Path

from models.vqvae.vqvae import VQVAE
from datasets.image_dataset import PairedGlyphImageDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_dir', type=str, default='data/target', help='Target images directory')
    parser.add_argument('--ref_dir', type=str, default='data/reference', help='Reference images directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize dataset and dataloader
    dataset = PairedGlyphImageDataset(args.tgt_dir, args.ref_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize VQ-VAE model
    vqvae_config = {
        'in_channels': 1,
        'base_channels': 64,
        'latent_dim': 2,
        'codebook_size': 64,
        'commitment_cost': 0.25
    }
    model = VQVAE(
        in_channels=vqvae_config['in_channels'],
        base_channels=vqvae_config['base_channels'],
        latent_dim=vqvae_config['latent_dim'],
        codebook_size=vqvae_config['codebook_size'],
        commitment_cost=vqvae_config['commitment_cost']
    ).to(device)
    
    # Initialize optimizer and scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for batch in pbar:
                tgt_imgs = batch['tgt_img'].to(device)
                ref_imgs = batch['ref_img'].to(device)
                
                # Forward pass for target images
                with autocast(device_type=device.type):
                    tgt_recon, tgt_vq_loss = model(tgt_imgs)
                    tgt_recon_loss = torch.nn.functional.mse_loss(tgt_recon, tgt_imgs)
                    
                    # Forward pass for reference images
                    ref_recon, ref_vq_loss = model(ref_imgs)
                    ref_recon_loss = torch.nn.functional.mse_loss(ref_recon, ref_imgs)
                    
                    # Calculate total loss
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
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'vq_loss': vq_loss.item()
                })
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        
        print(f'Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, VQ Loss: {avg_vq_loss:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = save_dir / 'vqvae.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')

if __name__ == '__main__':
    main()