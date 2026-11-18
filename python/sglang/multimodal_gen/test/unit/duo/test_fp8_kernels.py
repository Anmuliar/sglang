"""Keep Realization gate ordering and Intent's BF16 rounding when switching kernels."""

import pytest
import torch

from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache
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
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90 FP8 attention",
)
def test_fp8_attention_descales_and_visible_history_on_graph_replay() -> None:
    torch.manual_seed(42)
    query, keys, values = (
        torch.randn(1, rows, 7, 128, device="cuda").to(torch.float8_e4m3fn)
        for rows in (129, 513, 513)
    )
    scales = torch.rand(1, 3, 7, device="cuda") + 0.25

    def forward(length: int) -> torch.Tensor:
        return flash_attn_with_kvcache(
            query,
            keys[:, :length],
            values[:, :length],
            q_descale=scales[:, 0],
            k_descale=scales[:, 1],
            v_descale=scales[:, 2],
            causal=False,
            num_splits=1,
        )

    for length in (263, 513):
        q, k, v = (
            tensor.float().transpose(1, 2) * scales[:, index, :, None, None]
            for index, tensor in enumerate(
                (query, keys[:, :length], values[:, :length])
            )
        )
        expected = ((q @ k.transpose(-1, -2) * (128**-0.5)).softmax(-1) @ v).transpose(
            1, 2
        )
        actual = forward(length)
        assert actual.dtype == torch.bfloat16
        assert (actual.float() - expected).norm() / expected.norm() < 0.03

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward(513)
    query.copy_(query.float().neg().to(query.dtype))
    graph.replay()
    torch.testing.assert_close(captured, forward(513), rtol=0, atol=0)
