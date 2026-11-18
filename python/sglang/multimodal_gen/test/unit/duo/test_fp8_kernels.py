"""Keep Realization gate ordering and Intent's BF16 rounding when switching kernels."""

import os

import pytest
import torch

from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache
from sglang.multimodal_gen.runtime.models.dits.duo.fp8_execution import SM90FP8Attention
from sglang.multimodal_gen.runtime.models.dits.duo.fp8_kernels import rotate


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_inverse_rotation_writes_only_the_requested_output_view() -> None:
    """Direct output must preserve surrounding rows and noncontiguous head storage."""
    torch.manual_seed(42)
    hidden_states = torch.randn(19, 7, 128, device="cuda", dtype=torch.bfloat16)
    signs = torch.randint(0, 2, (128,), device="cuda").float() * 2 - 1
    centers = torch.randn(7, 128, device="cuda")
    expected = rotate(hidden_states, signs, center=centers, inverse=True)
    storage = torch.full((23, 9, 128), 17, device="cuda", dtype=torch.bfloat16)
    output = storage[2:21, 1:8]
    actual = rotate(hidden_states, signs, center=centers, inverse=True, out=output)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool((storage[:2] == 17).all())
    assert bool((storage[21:] == 17).all())
    assert bool((storage[:, (0, 8)] == 17).all())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rotate(hidden_states, signs, center=centers, inverse=True, out=output)
    hidden_states.add_(0.5)
    graph.replay()
    torch.testing.assert_close(
        output,
        rotate(hidden_states, signs, center=centers, inverse=True),
        rtol=0,
        atol=0,
    )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "DUO_REALIZATION_FP8_ATTENTION_LIBRARY" not in os.environ,
    reason="requires SM90 and DUO_REALIZATION_FP8_ATTENTION_LIBRARY",
)
def test_sm90_fp8_attention_uses_valid_lengths_and_refreshes_graph_inputs() -> None:
    """The optional FA3 backend must retain cropped KV lengths during graph replay."""
    torch.manual_seed(42)
    backend = SM90FP8Attention(os.environ["DUO_REALIZATION_FP8_ATTENTION_LIBRARY"])
    query = torch.randn(129, 7, 128, device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randn(1, 513, 7, 128, device="cuda").to(torch.float8_e4m3fn)
    values = torch.randn_like(keys, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.full((1, 3, 7), 0.5, device="cuda")
    for length in (263, 513):
        expected = flash_attn_with_kvcache(
            query[None],
            keys[:, :length],
            values[:, :length],
            q_descale=scales[:, 0],
            k_descale=scales[:, 1],
            v_descale=scales[:, 2],
            softmax_scale=128**-0.5,
            causal=False,
            num_splits=1,
        )[0]
        actual = backend.forward(
            query, keys[:, :length], values[:, :length], scales, 128**-0.5
        )
        relative_error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm()
        # Different FP8 probability tilings measured 1.7-2.0% relative error.
        assert relative_error < 0.03
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.forward(query, keys, values, scales, 128**-0.5)
    query.copy_(query.float().neg().to(query.dtype))
    graph.replay()
    torch.testing.assert_close(
        captured,
        backend.forward(query, keys, values, scales, 128**-0.5),
        rtol=0,
        atol=0,
    )
