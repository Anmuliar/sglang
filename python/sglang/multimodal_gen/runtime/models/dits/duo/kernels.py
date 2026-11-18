"""Rounding-preserving kernels for the reference Duo transformer."""

import logging

import torch
from diffusers.models.activations import SwiGLU
from torch import nn

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm as DiffusionRMSNorm

logger = logging.getLogger(__name__)


class SGLangH3SwiGLU(SwiGLU):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Diffusers packs value before gate; the SGLang kernel takes gate first.
        value, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return silu_and_mul_with_activation_rounding(torch.cat((gate, value), -1))


def use_realization_kernels(model: nn.Module) -> None:
    # Replace only loaded instances: checkpoint keys and causal KV ownership stay put.
    norm_count, activation_count = 0, 0
    for name, module in tuple(model.named_modules()):
        if isinstance(module, nn.RMSNorm) and module.weight.dtype == torch.bfloat16:
            norm = DiffusionRMSNorm(module.weight.numel(), eps=module.eps)
            norm.weight = module.weight
            model.set_submodule(name, norm)
            norm_count += 1
        elif isinstance(module, SwiGLU):
            module.__class__ = SGLangH3SwiGLU
            activation_count += 1
        else:
            pass
    logger.info(
        "Realization kernels: %s SGLang Diffusion RMSNorm and %s rounded SwiGLU",
        norm_count,
        activation_count,
    )
