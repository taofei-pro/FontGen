from __future__ import annotations

import torch


class DiffusionScheduler:
    """Scheduler placeholder for diffusion sampling."""

    def __init__(self, steps: int = 1000) -> None:
        self.steps = steps

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return x
