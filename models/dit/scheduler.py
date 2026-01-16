from __future__ import annotations

import torch


class DiffusionScheduler:
    """Basic linear beta scheduler for diffusion training."""

    def __init__(
        self,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ) -> None:
        self.steps = steps
        self.device = device or torch.device("cpu")
        self.betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sigmas = self.sqrt_one_minus_alphas_cumprod

    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alphas_cumprod) * x + torch.sqrt(1 - alphas_cumprod) * noise

    def get_timesteps(
        self,
        sampling_steps: int,
        schedule: str = "linear",
        rho: float = 7.0,
    ) -> torch.Tensor:
        """Return a descending list of timesteps for sampling."""
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive.")
        if schedule == "linear":
            return torch.linspace(
                self.steps - 1,
                0,
                sampling_steps,
                dtype=torch.long,
                device=self.device,
            )
        if schedule == "karras":
            sigmas = self.get_karras_sigmas(sampling_steps, rho=rho)
            timesteps = self.t_from_sigma(sigmas)
            return self._unique_descending(timesteps)
        raise ValueError(f"Unknown schedule: {schedule}")

    def ddim_step(
        self,
        x: torch.Tensor,
        t: int,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single DDIM update (eta=0)."""
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        pred_x0 = (x - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(
            1 - alpha_bar_prev
        ) * pred_noise
        return x_prev

    def ddpm_step(
        self,
        x: torch.Tensor,
        t: int,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single DDPM update with posterior variance."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar)
        mean = coef1 * (x - coef2 * pred_noise)

        if t == 0:
            return mean

        posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar)
        noise = torch.randn_like(x)
        return mean + torch.sqrt(posterior_variance) * noise

    def get_alpha_sigma_lambda(self, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_bar = self.alphas_cumprod[t]
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1 - alpha_bar)
        lam = torch.log(alpha) - torch.log(sigma)
        return alpha, sigma, lam

    def alpha_sigma_from_lambda(
        self, lam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute alpha/sigma from lambda value."""
        alpha = 1.0 / torch.sqrt(1.0 + torch.exp(-2 * lam))
        sigma = 1.0 / torch.sqrt(1.0 + torch.exp(2 * lam))
        return alpha, sigma

    def get_karras_sigmas(self, sampling_steps: int, rho: float = 7.0) -> torch.Tensor:
        """Karras sigma schedule for improved sampling stability."""
        sigma_max = self.sigmas[-1]
        sigma_min = self.sigmas[0]
        ramp = torch.linspace(0, 1, sampling_steps, device=self.device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def t_from_sigma(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Map sigmas to nearest discrete timestep indices."""
        sigmas = sigmas.to(self.device)
        timesteps = []
        for sigma in sigmas:
            idx = torch.argmin(torch.abs(self.sigmas - sigma)).item()
            timesteps.append(idx)
        return torch.tensor(timesteps, device=self.device, dtype=torch.long)

    def _unique_descending(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Ensure timesteps are strictly descending and unique."""
        seen = set()
        uniq = []
        for t in timesteps.tolist():
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        if uniq[-1] != 0:
            uniq.append(0)
        return torch.tensor(uniq, device=self.device, dtype=torch.long)

    def t_from_lambda(self, lam: torch.Tensor) -> torch.Tensor:
        """Map log-SNR (lambda) to nearest discrete timestep indices."""
        _, sigma = self.alpha_sigma_from_lambda(lam)
        return self.t_from_sigma(sigma)
