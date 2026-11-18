"""Validate the optional DCP adapter against the reference causal transformer."""

import pytest
import torch

pytest.importorskip("torchtitan.models.duo_realization.modules.model")

from torchtitan.models.duo_realization.modules.kv_cache import (
    LayerKV,
)

from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    AttentionKVWorkspace,
    ChunkedLayerKV,
    adapter_epilogue,
)


@pytest.mark.parametrize("batch_history_copies", [False, True])
@pytest.mark.parametrize("capacity_tokens", [0, 32])
def test_attention_workspace_preserves_history_across_denoise_and_eviction(
    batch_history_copies: bool,
    capacity_tokens: int,
) -> None:
    generator = torch.Generator().manual_seed(320)
    prefix = tuple(torch.randn(1, 5, 2, 4, generator=generator) for _ in range(2))
    history = LayerKV().append(prefix)
    workspace = AttentionKVWorkspace(
        capacity_tokens=capacity_tokens, batch_history_copies=batch_history_copies
    )
    saved_buffers: tuple[int, ...] | None = None
    saved_prefix = tuple(tensor.clone() for tensor in prefix)
    for step in range(4):
        current = tuple(
            torch.randn(1, 3, 2, 3, 4, generator=generator)[:, :, :, 1, :]
            for _ in range(2)
        )
        actual = workspace.concatenate(history, current)
        if capacity_tokens:
            pointers = tuple(value.data_ptr() for value in workspace.buffers)
            if saved_buffers is None:
                saved_buffers = pointers
            else:
                assert pointers == saved_buffers
        else:
            pass
        for expected, value in zip(history.concatenate(current), actual):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        if step == 1:
            history = history.append(tuple(tensor.clone() for tensor in current))
        elif step == 2:
            history = history.select_ranges([(0, 2), (5, 8)])
        else:
            pass
    for actual, expected in zip(prefix, saved_prefix):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    different_session = LayerKV().append(tuple(-tensor for tensor in prefix))
    for expected, actual in zip(
        different_session.concatenate(current),
        workspace.concatenate(different_session, current),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "frozen_enabled,trained_enabled", [(True, True), (False, True), (True, False)]
)
def test_adapter_epilogue_preserves_intermediate_rounding(
    frozen_enabled: bool, trained_enabled: bool, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    else:
        pass
    torch.manual_seed(963)
    base = torch.randn(576, 5376, device="cuda", dtype=dtype)
    frozen = torch.randn_like(base, dtype=torch.float32) if frozen_enabled else None
    trained = torch.randn_like(base, dtype=torch.float32) if trained_enabled else None
    expected = base
    if frozen is not None:
        expected = expected + (frozen * 0.3).to(base.dtype)
    else:
        pass
    if trained is not None:
        expected = (expected + trained * 0.7).to(base.dtype)
    else:
        pass
    actual = adapter_epilogue(base, frozen, trained, 0.3, 0.7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "ranges", [[], [(0, 2), (2, 5), (8, 12)], [(1, 3), (6, 11)], [(5, 8)], [(0, 12)]]
)
def test_chunked_kv_selection_matches_reference(ranges: list[tuple[int, int]]) -> None:
    generator = torch.Generator().manual_seed(923)
    chunks = tuple(
        tuple(torch.randn(1, length, 2, 4, generator=generator) for _ in range(2))
        for length in (5, 3, 4)
    )
    original = LayerKV(chunks)
    current = ChunkedLayerKV(chunks)
    expected = original.select_ranges(ranges)
    actual = current.select_ranges(ranges)
    for left, right in zip(expected.materialize(), actual.materialize()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    tail = tuple(torch.randn(1, 2, 2, 4, generator=generator) for _ in range(2))
    actual = actual.append(tail)
    expected = expected.append(tail)
    for left, right in zip(expected.materialize(), actual.materialize()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert isinstance(actual, ChunkedLayerKV)
    for retained in actual.chunks[:-1]:
        if retained[0].numel() and not any(
            retained[0] is source[0] for source in chunks
        ):
            assert all(
                retained[0].untyped_storage().data_ptr()
                != source[0].untyped_storage().data_ptr()
                for source in chunks
            )
        else:
            pass


@pytest.mark.accelerator
@torch.inference_mode()
def test_calibrated_fp8_history_eviction_and_replay() -> None:
    """Reused encodings must follow live chunks after eviction and graph replay."""
    from sglang.multimodal_gen.runtime.models.dits.duo.fp8_execution import (
        CalibratedFP8Workspace,
        FP8AttentionCalibration,
    )
    from sglang.multimodal_gen.runtime.models.dits.duo.fp8_kernels import rotate

    torch.manual_seed(713)
    device = torch.device("cuda")
    calibration = FP8AttentionCalibration(
        centers=torch.randn(1, 2, 2, 128, device=device) * 0.1,
        scales=torch.full((1, 3, 2), 0.03125, device=device),
        signs=(torch.randint(0, 2, (128,), device=device) * 2 - 1).float(),
    )
    chunks = [
        tuple(
            torch.randn(1, length, 2, 128, device=device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        for length in (3, 5, 7)
    ]
    current = torch.randn(4, 2, 128, device=device, dtype=torch.bfloat16)
    workspace = CalibratedFP8Workspace(calibration, 0, 32)
    history = LayerKV(tuple(chunks[:2]))
    workspace.prepare(history, len(current), chunks[0][0])
    assert workspace.encoded_history_tokens == 8
    workspace.prepare(history, len(current), chunks[0][0])
    assert workspace.encoded_history_tokens == 8
    history = LayerKV(tuple(chunks[1:]))
    workspace.prepare(history, len(current), chunks[0][0])
    assert workspace.encoded_history_tokens == 15
    assert len(workspace.encoded_chunks) == 2
    _, keys, values = workspace.encode(current, current, current)
    for side, actual in enumerate((keys, values)):
        original = torch.cat([chunks[1][side][0], chunks[2][side][0], current])
        reference = rotate(
            original, calibration.signs, center=calibration.centers[0, side]
        )
        expected = (
            (reference.float() / calibration.scales[0, side + 1, :, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(actual[0].float(), expected.float(), rtol=0, atol=0)
    pointer = workspace.buffers[0].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = workspace.encode(current, current, current)[0]
    current.add_(0.25)
    graph.replay()
    expected = workspace.encode(current, current, current)[0]
    torch.testing.assert_close(captured.float(), expected.float(), rtol=0, atol=0)
    assert workspace.buffers[0].data_ptr() == pointer
