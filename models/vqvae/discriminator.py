"""
PatchGAN Discriminator for VQ-GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for VQ-GAN.
    Discriminates between real and fake image patches.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Build discriminator layers
        layers = []
        
        # First layer
        layers.extend([
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Middle layers
        for i in range(1, num_layers):
            in_ch = base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2 ** i)
            
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final layer
        in_ch = base_channels * (2 ** (num_layers - 1))
        out_ch = base_channels * (2 ** num_layers)
        
        layers.extend([
            nn.Conv2d(in_ch, out_ch, 4, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, 1, 4, stride=1, padding=1)
        ])
        
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Discriminator output of shape (B, 1, H', W')
        """
        return self.discriminator(x)
    
    def compute_discriminator_loss(
        self, 
        real_logits: torch.Tensor, 
        fake_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discriminator loss using least squares GAN loss.
        
        Args:
            real_logits: Discriminator output for real images
            fake_logits: Discriminator output for fake images
            
        Returns:
            Discriminator loss
        """
        real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits))
        fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) * 0.5
    
    def compute_generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss using least squares GAN loss.
        
        Args:
            fake_logits: Discriminator output for fake images
            
        Returns:
            Generator loss
        """
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
