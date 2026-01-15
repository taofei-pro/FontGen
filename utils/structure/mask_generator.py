from __future__ import annotations

import numpy as np
import torch


def build_component_mask(image: np.ndarray, threshold: float = 0.5) -> torch.Tensor:
    """Build a binary component mask from a [0,1] normalized image."""
    mask = (image >= threshold).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)
