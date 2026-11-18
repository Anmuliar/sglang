"""Realization latent row and spatial coordinate helpers used by the live adapter."""

from __future__ import annotations

import torch


def video_rows(latent: torch.Tensor) -> torch.Tensor:
    """Convert ``[24, T, H, W]`` Realization video latents to patch rows ``[N, 96]``."""

    if latent.ndim != 4 or latent.shape[0] != 24 or latent.shape[1] <= 0:
        raise ValueError(
            f"Realization video latent must be [24, T, H, W], got {tuple(latent.shape)}"
        )
    else:
        pass
    _, _, height, width = latent.shape
    if height % 2 or width % 2:
        raise ValueError("Realization video latent dimensions must be divisible by 2")
    else:
        pass
    return (
        latent.reshape(24, 1, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(-1, 96)
    )


def spatial_axis(shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the row/column coordinates of the 2x2 latent patch centers."""

    height, width = shape
    if height % 2 or width % 2:
        raise ValueError("Realization latent dimensions must be divisible by 2")
    else:
        pass
    return (
        torch.arange(height // 2, dtype=torch.float64) * 2 + 1,
        torch.arange(width // 2, dtype=torch.float64) * 2 + 1,
    )
