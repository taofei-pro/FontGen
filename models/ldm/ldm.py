import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from models.vqvae.vqvae import VQVAE
from models.unet.unet import UNet


class TimeEmbedding(nn.Module):
    def __init__(self, time_pos_dim: int, time_emb_dim: int, time_steps: int, device: torch.device):
        super().__init__()
        self.time_steps = time_steps
        self.device = device
        
        # Create sinusoidal embeddings
        self.embedding = nn.Sequential(
            nn.Linear(time_pos_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Generate sinusoidal positional encodings
        self.register_buffer('positional_encodings', self._generate_positional_encodings(time_steps, time_pos_dim))
    
    def _generate_positional_encodings(self, time_steps: int, dim: int):
        encodings = torch.zeros(time_steps, dim, device=self.device)
        position = torch.arange(0, time_steps, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=self.device) * (-torch.log(torch.tensor(10000.0, device=self.device)) / dim))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        return encodings
    
    def forward(self, t: torch.Tensor):
        # t: [B] - time steps
        emb = self.positional_encodings[t].to(self.device)
        return self.embedding(emb)


class SigmoidScheduler:
    def __init__(self, noise_steps: int, device: torch.device):
        self.noise_steps = noise_steps
        self.device = device
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # Create beta schedule using sigmoid function
        self.betas = self._sigmoid_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Precompute noise schedule parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def _sigmoid_beta_schedule(self):
        t = torch.linspace(0, 1, self.noise_steps, device=self.device)
        beta = self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(10 * (t - 0.5))
        return beta
    
    def sample_timesteps(self, batch_size: int):
        return torch.randint(0, self.noise_steps, (batch_size,), device=self.device)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x, device=self.device)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def set_timesteps(self, num_steps: int):
        self.timesteps = torch.linspace(self.noise_steps - 1, 0, num_steps, device=self.device, dtype=torch.long)
        return self.timesteps
    
    def ddpm_step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        beta_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Compute x_prev
        model_mean = sqrt_recip_alphas_t * (x - beta_t * y / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x, device=self.device)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def ddim_step(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, y: torch.Tensor):
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev].reshape(-1, 1, 1, 1)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t_prev = self.sqrt_one_minus_alphas_cumprod[t_prev].reshape(-1, 1, 1, 1)
        
        # Compute x_prev
        numerator = x - sqrt_one_minus_alpha_cumprod_t * y
        denominator = sqrt_alpha_cumprod_t
        x0_pred = numerator / denominator
        
        sigma_t = torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
        noise = torch.randn_like(x, device=self.device)
        
        return sqrt_alpha_cumprod_t_prev * x0_pred + sqrt_one_minus_alpha_cumprod_t_prev * noise


class LDM(nn.Module):
    def __init__(self, vqvae_config: dict, ldm_config: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Initialize VQ-VAE
        self.vqvae = VQVAE(
            in_channels=vqvae_config['in_channels'],
            base_channels=vqvae_config['base_channels'],
            latent_dim=vqvae_config['latent_dim'],
            codebook_size=vqvae_config['codebook_size'],
            commitment_cost=vqvae_config['commitment_cost']
        ).to(device)
        
        # Initialize time embedding
        self.time_emb = TimeEmbedding(
            time_pos_dim=ldm_config['time_pos_dim'],
            time_emb_dim=ldm_config['time_emb_dim'],
            time_steps=ldm_config['time_steps'],
            device=device
        )
        
        # Initialize UNet
        self.unet = UNet(
            in_channels=vqvae_config['latent_dim'] * 2,  # Concatenate target and reference latents
            out_channels=vqvae_config['latent_dim'],
            base_channels=ldm_config['unet_base_channels'],
            time_emb_dim=ldm_config['time_emb_dim'],
            device=device
        )
        
        # Initialize scheduler
        self.scheduler = SigmoidScheduler(
            noise_steps=ldm_config['time_steps'],
            device=device
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def load_vqvae_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vqvae.load_state_dict(checkpoint)
        print(f"Loaded VQ-VAE checkpoint from {checkpoint_path}")
    
    def freeze_vqvae(self):
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.vqvae.eval()
    
    def forward(self, x: torch.Tensor, ref: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_emb(t)
        x_cat = torch.cat([x, ref], dim=1)
        noise_pred = self.unet(x_cat, t_emb)
        return noise_pred
    
    def encode_to_latent(self, images: torch.Tensor):
        return self.vqvae.encode(images)
    
    def decode_from_latent(self, latents: torch.Tensor):
        return self.vqvae.decode(latents)
    
    def train_step(self, batch: dict):
        tgt_imgs = batch['tgt_img'].to(self.device)
        ref_imgs = batch['ref_img'].to(self.device)
        
        batch_size = tgt_imgs.shape[0]
        t = self.scheduler.sample_timesteps(batch_size)
        
        tgt_latents = self.encode_to_latent(tgt_imgs)
        x_t, noise = self.scheduler.add_noise(tgt_latents, t)
        ref_latents = self.encode_to_latent(ref_imgs)
        
        noise_pred = self(x_t, ref_latents, t)
        loss = self.loss_fn(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def generate(self, ref_imgs: torch.Tensor, sample_steps: int = 50):
        self.eval()
        
        ref_latents = self.encode_to_latent(ref_imgs)
        batch_size, channels, height, width = ref_latents.shape
        
        # Sample random noise
        latents = torch.randn(batch_size, channels, height, width, device=self.device)
        
        # Denoise using DDIM
        self.scheduler.set_timesteps(sample_steps + 1)
        timesteps = self.scheduler.timesteps
        
        for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_tensor = torch.full((batch_size,), t, device=self.device)
            t_prev_tensor = torch.full((batch_size,), t_prev, device=self.device)
            
            noise_pred = self(latents, ref_latents, t_tensor)
            latents = self.scheduler.ddim_step(latents, t_tensor, t_prev_tensor, noise_pred)
        
        # Decode latents to images
        generated_imgs = self.decode_from_latent(latents)
        return generated_imgs