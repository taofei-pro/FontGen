from __future__ import annotations

import torch


def tile_infer(model, image: torch.Tensor, tile_size: int = 256) -> torch.Tensor:
    """Placeholder for tiled super-resolution inference."""
    _ = tile_size
    return model(image)
