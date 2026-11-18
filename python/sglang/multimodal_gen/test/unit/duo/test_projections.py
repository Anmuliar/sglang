"""Keep Realization gate ordering and Intent's BF16 rounding when switching kernels."""

import copy

import pytest
import torch
from diffusers.models.activations import SwiGLU

from sglang.multimodal_gen.runtime.models.dits.duo.kernels import (
    SGLangH3SwiGLU,
    use_realization_kernels,
)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 64])
def test_native_swiglu_preserves_loaded_projection_order(tokens: int) -> None:
    torch.manual_seed(42)
    reference_realization = SwiGLU(128, 256, bias=False).to("cuda", torch.bfloat16)
    native_realization = copy.deepcopy(reference_realization)
    native_realization.__class__ = SGLangH3SwiGLU
    hidden = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(
            native_realization(hidden), reference_realization(hidden), rtol=0, atol=0
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_realization_norm_preserves_mixed_dtype_output(
    weight_dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    reference = torch.nn.Sequential(
        torch.nn.RMSNorm(5120, eps=1e-5, device="cuda", dtype=weight_dtype)
    )
    reference[0].weight.data.uniform_(0.8, 1.2)
    native = copy.deepcopy(reference)
    use_realization_kernels(native)
    hidden = torch.randn(1, 64, 5120, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = reference(hidden)
        actual = native(hidden)
    assert actual.dtype == expected.dtype
    relative_error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm()
    assert relative_error < 1e-3
