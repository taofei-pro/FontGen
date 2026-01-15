from __future__ import annotations

import numpy as np
import torch
from skimage import feature


def build_edge_map(image: np.ndarray, sigma: float = 1.0) -> torch.Tensor:
    """Build an edge map using Canny on a [0,1] normalized image."""
    edges = feature.canny(image, sigma=sigma)
    return torch.from_numpy(edges.astype(np.float32)).unsqueeze(0)
