"""Validate the optional DCP adapter against the reference causal transformer."""

import os
import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchtitan.models.duo_realization.modules.model")

from torchtitan.models.duo_realization.modules.kv_cache import (
    KVCache,
    LayerKV,
    commit_rows,
)

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.models.dits.duo.modelopt import (
    ModelOptFp8DirectMethod,
    modelopt_tensor,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_dcp import (
    attention_regions,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    NativeDCPOptions,
)
from sglang.multimodal_gen.runtime.models.dits.duo.ring import (
    RingKVCache,
    ring_attention,
    ring_plan,
)


def test_modelopt_qkv_reorders_scales_with_their_fp8_rows() -> None:
    weight = torch.arange(96).reshape(48, 2).to(torch.float8_e4m3fn)
    scales = torch.arange(48, dtype=torch.float32) + 1
    layer = torch.nn.Identity()
    actual_weight = modelopt_tensor(
        "blocks.0.attn.qkv_proj.weight", weight, layer, heads=4, head_dim=4
    )
    actual_scales = modelopt_tensor(
        "blocks.0.attn.qkv_proj.weight_scale", scales, layer, heads=4, head_dim=4
    )
    rows = torch.arange(48).reshape(4, 3, 4).permute(1, 0, 2).reshape(-1)
    assert actual_weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        actual_weight.view(torch.uint8), weight.view(torch.uint8)[rows], rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scales, scales[rows], rtol=0, atol=0)


@pytest.mark.parametrize("degree", [4, 8])
def test_modelopt_bias_sharding_keeps_one_dimensional_values(degree: int) -> None:
    # Layout conversion only reads rank metadata; no process group is needed.
    layer = object.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.tp_size = degree
    layer.tp_rank = degree - 1
    bias = torch.arange(64, dtype=torch.bfloat16)
    actual = modelopt_tensor(
        "blocks.0.adaln_proj.linear.bias", bias, layer, heads=8, head_dim=4
    )
    torch.testing.assert_close(actual, bias[64 - 64 // degree :], rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_modelopt_direct_kernels_match_default_and_refresh_graph_inputs() -> None:
    with torch.cuda.device(0):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        )
        reference = Fp8LinearMethod(config)
        if not reference.cutlass_fp8_supported or reference.use_marlin:
            pytest.skip("requires native CUTLASS FP8")
        else:
            pass
        direct = ModelOptFp8DirectMethod(config)
        layer = torch.nn.Linear(
            256, 256, bias=False, device="cuda:0", dtype=torch.bfloat16
        )
        weight = layer.weight.detach()
        scale = weight.float().abs().amax(1).clamp_min(1e-12) / 448
        layer.weight = torch.nn.Parameter(
            (weight.float() / scale[:, None]).to(torch.float8_e4m3fn).t(),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(scale[None], requires_grad=False)
        layer.input_scale = None
        value = torch.randn(64, 256, device="cuda:0", dtype=torch.bfloat16)
        value[0].zero_()
        value[1].fill_(1e-20)
        torch.testing.assert_close(
            direct.apply(layer, value), reference.apply(layer, value), rtol=0, atol=0
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = direct.apply(layer, value)
        value.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(
            output, reference.apply(layer, value), rtol=0, atol=0
        )


def run_ring_contract(rank: int, degree: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(degree))
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=degree,
        ulysses_degree=1,
        ring_degree=degree,
        distributed_init_method=rendezvous,
    )
    torch.manual_seed(1729)
    full_cache = None
    cache = None
    with torch.inference_mode():
        for block in range(4):
            if block == 3:
                full_cache = full_cache.evict(2)
                cache = cache.evict(2)
                assert cache.length == full_cache.length
            else:
                pass
            # Deliberately uneven ownership, disjoint visibility and padding.
            groups = (
                (
                    dict(start=0, end=3, kind="reference", block=-1, sample=0),
                    dict(start=3, end=5, kind="shared", block=-1, sample=0),
                )
                if block == 0
                else ()
            )
            offset = 5 if block == 0 else 0
            groups += (
                dict(start=offset, end=offset + 3, kind="text", block=block, sample=0),
                dict(
                    start=offset + 3,
                    end=29,
                    kind="noisy" if block == 0 else "clean",
                    block=block,
                    sample=0,
                ),
                dict(start=29, end=32, kind="padding", block=block, sample=0),
            )
            q, k, v = [
                torch.randn(32, 2, 64, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            past = 0 if cache is None else cache.length
            key_groups = (() if cache is None else cache.groups) + tuple(
                dict(g, start=g["start"] + past, end=g["end"] + past) for g in groups
            )
            regions = attention_regions(groups, key_groups)
            plan = ring_plan(regions, groups, 32, cache, commit=True, device=q.device)
            local_q, local_k, local_v = [t.chunk(degree)[rank] for t in (q, k, v)]
            retained = (
                local_k[plan.retained].unsqueeze(0),
                local_v[plan.retained].unsqueeze(0),
            )
            local_layer = LayerKV() if cache is None else cache.layers[0]
            cached_k, cached_v = local_layer.concatenate(
                (local_k.unsqueeze(0), local_v.unsqueeze(0))
            )
            actual = ring_attention(local_q, cached_k, cached_v, plan, 64**-0.5)
            full_layer = LayerKV() if full_cache is None else full_cache.layers[0]
            full_k, full_v = full_layer.concatenate((k.unsqueeze(0), v.unsqueeze(0)))
            expected = torch.empty_like(q)
            for region in regions:
                keys, values = [
                    torch.cat([t[:, lo:hi] for lo, hi in region.keys], dim=1)
                    for t in (full_k, full_v)
                ]
                expected[region.start : region.end] = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q[region.start : region.end].unsqueeze(0).transpose(1, 2),
                        keys.transpose(1, 2),
                        values.transpose(1, 2),
                    ).transpose(1, 2)[0]
                )
            torch.testing.assert_close(
                actual, expected.chunk(degree)[rank], rtol=0.02, atol=0.02
            )
            rows, additions = commit_rows(groups, q.device, past)
            groups = (() if cache is None else cache.groups) + additions
            cache = RingKVCache(
                (local_layer.append(retained),), groups, plan.shards, rank
            )
            full_cache = KVCache(
                (full_layer.append((k[rows].unsqueeze(0), v[rows].unsqueeze(0))),),
                groups,
            )
            ids = [i for lo, hi in cache.shards[rank] for i in range(lo, hi)]
            for local, full in zip(
                cache.layers[0].materialize(), full_cache.layers[0].materialize()
            ):
                torch.testing.assert_close(local, full[:, ids], rtol=0, atol=0)
            assert cache.nbytes == len(ids) * 2 * 2 * 64 * 2
    torch.distributed.destroy_process_group()


@pytest.mark.accelerator
@pytest.mark.parametrize("degree", [2, 4])
def test_ring_matches_dense_with_history_and_eviction(degree: int) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    else:
        pass
    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            run_ring_contract,
            args=(degree, (Path(directory) / "ring").as_uri()),
            nprocs=degree,
            join=True,
        )


@pytest.mark.parametrize("packed", [False, True])
def test_fixed_adapter_merge_preserves_weight_and_is_idempotent(packed: bool) -> None:
    from sglang.multimodal_gen.runtime.models.dits.duo.native_dcp import (
        AdapterDelta,
        DCPLinear,
        NativeDCPRealization,
    )

    torch.manual_seed(19)
    model = NativeDCPRealization.__new__(NativeDCPRealization)
    torch.nn.Module.__init__(model)
    base = torch.nn.Linear(16, 24, bias=False, dtype=torch.bfloat16)
    a = torch.randn((3, 4, 16) if packed else (4, 16))
    b = torch.randn((3, 8, 4) if packed else (24, 4))
    delta = (b @ a).reshape(24, 16) * 0.125
    expected = (base.weight.float() + delta).to(torch.bfloat16)
    model.projection = DCPLinear(base, None, AdapterDelta(a, b, 0.125))
    options = NativeDCPOptions(merge_fixed_lora=True)
    model.prepare_fixed_weights(options)
    assert model.projection is base
    torch.testing.assert_close(base.weight, expected, rtol=0, atol=0)
    model.prepare_fixed_weights(options)
    torch.testing.assert_close(base.weight, expected, rtol=0, atol=0)
    inputs = torch.randn(3, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(model.projection(inputs), inputs @ expected.T)
