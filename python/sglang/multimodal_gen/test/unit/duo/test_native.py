"""Validate the optional DCP adapter against the reference causal transformer."""

import os
import tempfile
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchtitan.models.duo_realization.modules.model")

from data.dataloader_realization import RealizationSequence
from data.dataloader_realization_sequence import block_seed
from torchtitan.models.duo_realization.modules.decfg import install_adapter
from torchtitan.models.duo_realization.modules.kv_cache import (
    KVCache,
)
from torchtitan.models.duo_realization.modules.model import RealizationTransformer
from torchtitan.models.duo_realization.modules.weights import configure_trainables
from torchtitan.models.duo_realization.pipelines.inference import sample_sequence

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_dcp import (
    native_dcp_from_reference,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    ChunkedLayerKV,
    NativeDCPOptions,
)
from sglang.multimodal_gen.runtime.pipelines.duo.session import (
    NativeBudgetMediaSession,
    TokenWindowMediaSession,
)


def run_native_dcp_contract(
    rank: int,
    degree: int,
    rendezvous: str,
    graph: bool = False,
    tensor_parallel: bool = False,
) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(degree))
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=degree if tensor_parallel or not graph else 1,
        sp_size=degree if graph and not tensor_parallel else 1,
        ulysses_degree=degree if graph and not tensor_parallel else 1,
        distributed_init_method=rendezvous,
    )
    torch.manual_seed(42)
    reference = RealizationTransformer(
        hidden_size=128,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_dim=256,
        num_layers=2,
        num_refiner_layers=1,
        text_dim=64,
        time_embed_dim=64,
        time_embed_hidden_dim=128,
        freq_dim=32,
        rope_freq_dim=8,
    )
    configure_trainables(
        reference, "lora", rank=4, alpha=4, trainable=False, role_embedding=True
    )
    install_adapter(reference, 4)
    for name, parameter in reference.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
        else:
            pass
    reference.eval().cuda()

    def group(start: int, end: int, kind: str, block: int = -1) -> dict:
        return dict(start=start, end=end, kind=kind, block=block, sample=0)

    groups = (
        group(0, 4, "reference"),
        group(4, 7, "shared"),
        group(7, 9, "text", 0),
        group(9, 19, "noisy", 0),
        group(19, 32, "padding"),
    )
    video_indices = torch.tensor([0, 1, 2, 3, *range(9, 17)], device="cuda")
    audio_indices = torch.tensor([17, 18], device="cuda")
    text_indices = torch.arange(4, 9, device="cuda")
    text = torch.randn(1, 5, 64, device="cuda", dtype=torch.bfloat16)
    tags = torch.zeros(32, dtype=torch.long, device="cuda")
    tags[text_indices], tags[audio_indices] = 1, 2
    time_ids = torch.zeros_like(tags)
    time_ids[9:19] = 1
    sequence = RealizationSequence(
        video=torch.randn(1, 12, 96, device="cuda"),
        audio=torch.randn(1, 2, 32, device="cuda"),
        text=text,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        positions=torch.arange(32, device="cuda")[:, None].expand(-1, 3),
        tags=tags,
        roles=torch.full_like(tags, -1),
        time_ids=time_ids,
        video_weights=torch.ones(12, device="cuda"),
        audio_weights=torch.ones(2, device="cuda"),
        groups=groups,
        text_groups=(),
        samples=(),
        seeds=(),
        refiner_text=text,
        refiner_groups=(group(0, 3, "shared"), group(3, 5, "text", 0)),
        text_selection=torch.arange(5, device="cuda"),
    )
    times = torch.tensor([1.0, 0.5], device="cuda")
    with torch.inference_mode():
        expected, reference_cache = reference(sequence, times, commit=True)
        native = native_dcp_from_reference(
            reference, device=torch.device("cuda"), use_flash_attention=False
        )
        actual, cache = native(sequence, times, commit=True)
        for left, right in zip(expected, actual):
            assert (left.float() - right.float()).norm() / left.float().norm() < 0.01
        for local, full in zip(cache.layers, reference_cache.layers):
            for local_tensor, full_tensor in zip(
                local.materialize(), full.materialize()
            ):
                expected_shard = full_tensor.chunk(degree, dim=2)[rank]
                assert local_tensor.shape == expected_shard.shape
                error = (local_tensor.float() - expected_shard.float()).norm()
                assert error / expected_shard.float().norm() < 0.01
        assert cache.nbytes * degree == reference_cache.nbytes
        assert cache.groups == reference_cache.groups
        assert cache.length == 7  # No noisy media or current plan is committed.
        changed_video = sequence.video.clone()
        changed_video[:, 4:] += 2.0
        changed = sequence.with_media(changed_video, sequence.audio + 2.0)
        _, changed_cache = native(changed, times, commit=True)
        for left, right in zip(cache.layers, changed_cache.layers):
            for cached, changed_cached in zip(left.materialize(), right.materialize()):
                torch.testing.assert_close(cached, changed_cached, rtol=0, atol=0)
        projections = native.timestep_projections(times)
        cached, _ = native(sequence, times, timestep_projections=projections)
        uncached, _ = native(sequence, times)
        for expected_tensor, actual_tensor in zip(uncached, cached):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        native.use_flash_attention = True
        flash, _ = native(sequence, times, commit=True)
        for left, right in zip(actual, flash):
            assert (left.float() - right.float()).norm() / left.float().norm() < 0.01

        if degree == 1 and not graph:
            native.options.cache_timestep_projections = False
            sample_input = replace(sequence, seeds=(123,))
            session = NativeBudgetMediaSession(
                native,
                {"sample_id": "finite-check", "codec": {"mode": "duo_5_17"}},
                Namespace(stage="dmd", steps=4, flow_shift=5.0, seed=123),
                torch.device("cuda"),
            )
            session.kv = cache
            expected_values, expected_metrics = sample_sequence(
                native,
                sample_input,
                stage="dmd",
                steps=4,
                shift=5.0,
                seed=block_seed(123, "finite-check", 0),
                history=cache,
            )
            actual_values, actual_metrics = session.sample_dmd(sample_input, 0)
            for expected_tensor, actual_tensor in zip(expected_values, actual_values):
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, rtol=0, atol=0
                )
            assert actual_metrics == expected_metrics

            def nonfinite_prediction(module, inputs, output):
                prediction, kv = output
                return (
                    tuple(torch.full_like(value, float("nan")) for value in prediction),
                    kv,
                )

            handle = native.register_forward_hook(nonfinite_prediction)
            try:
                with pytest.raises(FloatingPointError, match="sampling step 0"):
                    session.sample_dmd(sample_input, 0)
            finally:
                handle.remove()
            assert session.kv is cache
        else:
            pass

        if graph:
            native.configure_execution(
                NativeDCPOptions(
                    enable_cuda_graph=True,
                    batch_history_copies=True,
                    reuse_adapter_input_cast=True,
                    direct_attention_output=True,
                )
            )
            clean_projection = native.timestep_projections(torch.ones_like(times))
            saved_commits = []
            for iteration in range(3):
                changed = replace(
                    sequence,
                    video=sequence.video + iteration * 0.1,
                    audio=sequence.audio - iteration * 0.1,
                    refiner_text=sequence.refiner_text + iteration * 0.2,
                    positions=sequence.positions + iteration * 3,
                )
                for commit, projection in (
                    (False, native.timestep_projections(times * (iteration + 1) / 3)),
                    (True, clean_projection),
                ):
                    clean = (
                        replace(
                            changed,
                            groups=tuple(
                                (
                                    dict(group, kind="clean")
                                    if group["kind"] == "noisy"
                                    else group
                                )
                                for group in changed.groups
                            ),
                        )
                        if commit
                        else changed
                    )
                    native.options.enable_cuda_graph = False
                    expected, expected_cache = native(
                        clean,
                        times,
                        cache=cache,
                        commit=commit,
                        timestep_projections=projection,
                    )
                    native.options.enable_cuda_graph = True
                    actual, actual_cache = native(
                        clean,
                        times,
                        cache=cache,
                        commit=commit,
                        timestep_projections=projection,
                    )
                    for left, right in zip(expected, actual):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    for old_cache, saved_layers in saved_commits:
                        for old_layer, saved_values in zip(
                            old_cache.layers, saved_layers
                        ):
                            for old_value, saved_value in zip(
                                old_layer.chunks[-1], saved_values
                            ):
                                torch.testing.assert_close(
                                    old_value, saved_value, rtol=0, atol=0
                                )
                    if commit:
                        saved_commits.append(
                            (
                                actual_cache,
                                [
                                    tuple(value.clone() for value in layer.chunks[-1])
                                    for layer in actual_cache.layers
                                ],
                            )
                        )
                        for left, right in zip(
                            expected_cache.layers, actual_cache.layers
                        ):
                            for before, after in zip(
                                left.materialize(), right.materialize()
                            ):
                                torch.testing.assert_close(
                                    before, after, rtol=0, atol=0
                                )
                        # Keep shape, replace the contents and backing storage of history.
                        cache = KVCache(
                            tuple(
                                ChunkedLayerKV(
                                    tuple(
                                        tuple(value + 0.01 for value in chunk)
                                        for chunk in layer.chunks
                                    )
                                )
                                for layer in cache.layers
                            ),
                            cache.groups,
                        )
            assert native.graph_captures == 2
            assert native.graph_replays == 6
            native.clear_execution_cache()
            torch.distributed.destroy_process_group()
        else:
            pass


@pytest.mark.accelerator
@pytest.mark.parametrize("degree,tensor_parallel", [(1, False), (4, False), (4, True)])
def test_native_dcp_graph_refreshes_inputs_and_history(
    degree: int, tensor_parallel: bool
) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    else:
        pass
    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            run_native_dcp_contract,
            args=(degree, (Path(directory) / "graph").as_uri(), True, tensor_parallel),
            nprocs=degree,
            join=True,
        )


@pytest.mark.accelerator
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_native_dcp_preserves_adapters_and_causal_prefix(degree: int) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    else:
        pass
    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            run_native_dcp_contract,
            args=(degree, (Path(directory) / "tp").as_uri()),
            nprocs=degree,
            join=True,
        )


@pytest.mark.parametrize(
    "window_chunks, expected_starts",
    [(0, [0, 0, 0, 2, 2, 3]), (2, [0, 0, 1, 2, 3, 4])],
)
def test_token_window_evicts_whole_blocks_without_rewriting_retained_kv(
    window_chunks: int, expected_starts: list[int]
) -> None:
    """Shorter plans cannot resurrect evicted history or change retained K/V."""
    session = TokenWindowMediaSession(
        torch.nn.Identity(),
        {"codec": {"mode": "duo_5_17"}},
        Namespace(),
        torch.device("cpu"),
        7400,
        reference_window_chunks=window_chunks,
    )
    session.index = {
        "shared_rows": 590,
        "blocks": [
            {"video_rows": 2160, "stereo_audio_rows": 58, "packed_rows": 4436 + count}
            for count in (1, 1, 1, 1000, 1, 1)
        ],
    }
    prefix = torch.randn(1, 590, 1, 2)
    cache = KVCache(
        (ChunkedLayerKV(((prefix, prefix.clone()),)),),
        ({"kind": "reference", "start": 0, "end": 590, "block": -1},),
    )
    for block, expected_start in enumerate(expected_starts):
        start = session.window_start(block)
        assert start == expected_start
        previous_chunks = cache.layers[0].chunks
        cache = cache.evict(start)
        session.start = start
        assert cache.layers[0].chunks[0][0].data_ptr() == prefix.data_ptr()
        for retained in cache.layers[0].chunks:
            assert any(
                retained[0] is old[0] and retained[1] is old[1]
                for old in previous_chunks
            )
        current = torch.randn(1, 2218, 1, 2)
        cache = KVCache(
            (cache.layers[0].append((current, current.clone())),),
            (
                *cache.groups,
                {
                    "kind": "clean",
                    "start": cache.length,
                    "end": cache.length + 2218,
                    "block": block,
                },
            ),
        )
        plan_tokens = session.index["blocks"][block]["packed_rows"] - 4436
        assert cache.length + plan_tokens <= session.window_tokens
        assert [group["block"] for group in cache.groups[1:]] == list(
            range(start, block + 1)
        )
    session.window_tokens = 2808
    with pytest.raises(ValueError, match="cannot fit"):
        session.window_start(5)


def test_clean_commit_matches_reference_full_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = torch.randn(1, 8, 96)
    audio = torch.randn(1, 2, 32)
    text = torch.randn(1, 2, 64)
    groups = (
        dict(sample=0, block=3, kind="text", start=0, end=2),
        dict(sample=0, block=3, kind="noisy", start=2, end=12),
        dict(sample=-1, block=-1, kind="padding", start=12, end=128),
    )
    positions = torch.zeros(128, 3)
    positions[:12] = torch.randn(12, 3)
    tags = torch.zeros(128, dtype=torch.long)
    tags[2:12] = 1
    time_ids = tags.clone()
    sequence = RealizationSequence(
        video=video,
        audio=audio,
        text=text,
        video_indices=torch.arange(2, 10),
        audio_indices=torch.arange(10, 12),
        text_indices=torch.arange(2),
        positions=positions,
        tags=tags,
        roles=torch.full((128,), -1),
        time_ids=time_ids,
        video_weights=torch.ones(8),
        audio_weights=torch.ones(2),
        groups=groups,
        text_groups=(groups[0],),
        samples=(),
        seeds=(),
        refiner_text=text,
        refiner_groups=(groups[0],),
        text_selection=torch.arange(2),
    )
    values = (video + 1, audio - 1)
    expected = sequence.with_media(*values).select(
        lambda group: True, rename={"noisy": "clean"}
    )
    expected = replace(expected, time_ids=torch.zeros_like(expected.time_ids))
    session = object.__new__(NativeBudgetMediaSession)
    session.device = torch.device("cpu")
    session.clean_time = 1.0
    session.next_block = 3
    session.kv = KVCache((), ())

    def model(
        actual: RealizationSequence,
        times: torch.Tensor,
        *,
        cache: KVCache,
        commit: bool,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], KVCache]:
        assert commit and cache is session.kv
        torch.testing.assert_close(times, torch.ones(2), rtol=0, atol=0)
        for name, expected_value in vars(expected).items():
            actual_value = vars(actual)[name]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value
        return values, cache

    session.model = model
    monkeypatch.setattr(session, "measure", lambda name: nullcontext())
    session.commit(sequence, values, {})
    assert session.next_block == 4
    assert sequence.groups[1]["kind"] == "noisy"
    torch.testing.assert_close(sequence.video, video, rtol=0, atol=0)
