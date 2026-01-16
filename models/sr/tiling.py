from __future__ import annotations

import torch


def tile_infer(
    model,
    image: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 32,
) -> torch.Tensor:
    """Run tiled super-resolution inference with simple blending."""
    if image.dim() != 4:
        raise ValueError(f"Expected BCHW image tensor, got {image.shape}")
    if tile_size <= overlap:
        raise ValueError("tile_size must be greater than overlap.")

    scale = getattr(model, "scale", 1)
    b, c, h, w = image.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((b, c, out_h, out_w), device=image.device, dtype=image.dtype)
    weight = torch.zeros_like(output)

    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[:, :, y:y_end, x:x_end]
            sr_tile = model(tile)

            oy, ox = y * scale, x * scale
            oy_end, ox_end = oy + sr_tile.shape[2], ox + sr_tile.shape[3]

            output[:, :, oy:oy_end, ox:ox_end] += sr_tile
            weight[:, :, oy:oy_end, ox:ox_end] += 1

    return output / weight.clamp(min=1)
