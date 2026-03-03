import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
from pathlib import Path

from models.ldm.ldm import LDM
from datasets.image_dataset import PairedGlyphImageDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_dir', type=str, default='data/target', help='Target images directory')
    parser.add_argument('--ref_dir', type=str, default='data/reference', help='Reference images directory')
    parser.add_argument('--vqvae_checkpoint', type=str, default='checkpoints/vqvae.pth', help='VQ-VAE checkpoint path')
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
    
    # Load VQ-VAE checkpoint and freeze it
    model.load_vqvae_checkpoint(args.vqvae_checkpoint)
    model.freeze_vqvae()
    
    # Initialize optimizer and scaler
    optimizer = optim.Adam(model.unet.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for batch in pbar:
                # Forward pass
                with autocast(device_type=device.type):
                    loss = model.train_step(batch)
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        print(f'Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = save_dir / 'ldm.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')

if __name__ == '__main__':
    main()