"""Native Diffusion Realization backbone with the trained frame-v5 causal contract.

The reference package supplies sequence metadata and the DMD sampler. All model
projections, transformer blocks and modulation kernels belong to Diffusion.
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import torch
from data.dataloader_realization import RealizationSequence
from peft.tuners.lora.layer import Linear as PeftLinear
from torch import nn
from torch.nn import functional as F
from torchtitan.models.duo_realization.modules.adaln_timestep import (
    build_adaln_timestep_indices,
)
from torchtitan.models.duo_realization.modules.decfg import DeCFGLinear
from torchtitan.models.duo_realization.modules.kv_cache import (
    KVCache,
    LayerKV,
    commit_rows,
)
from torchtitan.models.duo_realization.modules.masks import visible
from torchtitan.models.duo_realization.modules.model import RealizationTransformer

from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_sp_group,
    get_tp_group,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import _compute_lora_delta
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_packed_qkv,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.duo.fp8_execution import (
    CalibratedFP8Workspace,
    FP8AttentionCalibration,
)
from sglang.multimodal_gen.runtime.models.dits.duo.fp8_kernels import rotate
from sglang.multimodal_gen.runtime.models.dits.duo.model import apply_attention_rope
from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    AttentionKVWorkspace,
    ChunkedLayerKV,
    NativeDCPOptions,
    adapter_epilogue,
)
from sglang.multimodal_gen.runtime.models.dits.duo.ring import (
    AttentionRegion,
    RingKVCache,
    RingPlan,
    ring_attention,
    ring_plan,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
    _apply_qk_norm,
    _diffusers_h3_checkpoint,
    _modulate_gate,
    _modulate_rmsnorm_scale_shift,
    _rope_cos_sin_cache,
)


class AdapterDelta(nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, scale: float) -> None:
        super().__init__()
        self.a = nn.Parameter(a, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _compute_lora_delta(value.float(), self.a, self.b) * self.scale


class DCPLinear(nn.Module):
    """Keep DeCFG then PEFT rounding while using Diffusion's fused base GEMM."""

    def __init__(
        self,
        base: LinearBase,
        frozen: AdapterDelta | None,
        trained: AdapterDelta | None,
    ) -> None:
        super().__init__()
        self.fuse_epilogue = True
        self.reuse_adapter_input_cast = False
        self.base = base
        self.frozen = frozen
        self.trained = trained
        self.reduce_output = isinstance(base, RowParallelLinear) and base.tp_size > 1
        self.gather_output = (
            isinstance(base, ColumnParallelLinear) and base.gather_output
        )
        if self.reduce_output:
            base.reduce_results = False
        elif self.gather_output:
            base.gather_output = False
        else:
            pass

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        result, bias = self.base(value)
        assert bias is None
        if self.fuse_epilogue:
            adapter_input = (
                value.float()
                if self.reuse_adapter_input_cast
                and (self.frozen is not None or self.trained is not None)
                else None
            )
            frozen = (
                None
                if self.frozen is None
                else _compute_lora_delta(
                    value.float() if adapter_input is None else adapter_input,
                    self.frozen.a,
                    self.frozen.b,
                )
            )
            trained = (
                None
                if self.trained is None
                else _compute_lora_delta(
                    value.float() if adapter_input is None else adapter_input,
                    self.trained.a,
                    self.trained.b,
                )
            )
            result = adapter_epilogue(
                result,
                frozen,
                trained,
                1.0 if self.frozen is None else self.frozen.scale,
                1.0 if self.trained is None else self.trained.scale,
            )
        else:
            dtype = result.dtype
            if self.frozen is not None:
                result = result + self.frozen(value).to(dtype)
            else:
                pass
            if self.trained is not None:
                result = (result + self.trained(value)).to(dtype)
            else:
                pass
        if self.reduce_output:
            result = tensor_model_parallel_all_reduce(
                result, tp_group=self.base.tp_group
            )
        elif self.gather_output:
            result = tensor_model_parallel_all_gather(
                result, tp_group=self.base.tp_group
            )
        else:
            pass
        return result, None


def shard_column(value: torch.Tensor, layer: ColumnParallelLinear) -> torch.Tensor:
    """Shard each logical projection in an already converted native tensor."""
    if isinstance(layer, MergedColumnParallelLinear) and value.ndim == 2:
        pieces = value.split(layer.output_sizes, dim=0)
        return torch.cat(
            [piece.chunk(layer.tp_size, dim=0)[layer.tp_rank] for piece in pieces]
        )
    else:
        return value.chunk(layer.tp_size, dim=-2)[layer.tp_rank].contiguous()


def shard_adapter(adapter: AdapterDelta | None, layer: LinearBase) -> None:
    if adapter is None:
        return
    elif isinstance(layer, ColumnParallelLinear):
        adapter.b = nn.Parameter(shard_column(adapter.b, layer), requires_grad=False)
    elif isinstance(layer, RowParallelLinear):
        adapter.a = nn.Parameter(
            adapter.a.chunk(layer.tp_size, dim=-1)[layer.tp_rank].contiguous(),
            requires_grad=False,
        )
    else:
        pass


def adapter_delta(
    sources: list[nn.Linear | PeftLinear], *, frozen: bool, swap_mlp: bool
) -> AdapterDelta | None:
    pairs, scales = [], []
    for source in sources:
        base = source.get_base_layer() if isinstance(source, PeftLinear) else source
        if frozen and isinstance(base, DeCFGLinear):
            pairs.append((base.decfg.a.weight, base.decfg.b.weight))
            scales.append(1.0)
        elif not frozen and isinstance(source, PeftLinear):
            assert source.active_adapters == ["default"] and not source.merged
            pairs.append(
                (source.lora_A["default"].weight, source.lora_B["default"].weight)
            )
            scales.append(source.scaling["default"])
        else:
            pass
    if not pairs:
        return None
    else:
        assert len(pairs) == len(sources) and len(set(scales)) == 1
    if len(pairs) == 1:
        a, b = pairs[0]
        if swap_mlp:
            value, gate = b.chunk(2, dim=0)
            b = torch.cat((gate, value))
        else:
            pass
    else:
        a = torch.stack([pair[0] for pair in pairs])
        b = torch.stack([pair[1] for pair in pairs])
    return AdapterDelta(a, b, scales[0])


def attention_regions(
    query_groups: tuple[dict[str, str | int], ...],
    key_groups: tuple[dict[str, str | int], ...],
    *,
    refiner: bool = False,
) -> tuple[AttentionRegion, ...]:
    regions: list[AttentionRegion] = []
    for query in query_groups:
        keys: list[tuple[int, int]] = []
        for key in key_groups:
            if visible(query, key, refiner=refiner):
                if keys and keys[-1][1] == key["start"]:
                    keys[-1] = (keys[-1][0], key["end"])
                else:
                    keys.append((key["start"], key["end"]))
            else:
                pass
        assert keys
        if (
            regions
            and regions[-1].end == query["start"]
            and regions[-1].keys == tuple(keys)
        ):
            regions[-1] = AttentionRegion(regions[-1].start, query["end"], tuple(keys))
        else:
            regions.append(AttentionRegion(query["start"], query["end"], tuple(keys)))
    return tuple(regions)


@dataclass(kw_only=True)
class TimestepProjections:
    blocks: tuple[tuple[torch.Tensor, ...], ...]
    final: tuple[torch.Tensor, torch.Tensor]
    packed_blocks: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class DCPBlockGraphKey:
    commit: bool
    history_tokens: int
    regions: tuple[AttentionRegion, ...]
    plan_regions: tuple[AttentionRegion, ...]
    plan_keys: tuple[tuple[AttentionRegion, ...], ...]
    input_shapes: tuple[tuple[torch.Size, torch.dtype], ...]
    workspace_pointers: tuple[int, ...]


@dataclass(kw_only=True)
class CapturedDCPBlocks:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    retained: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    plans: list[dict[AttentionRegion, torch.Tensor]] | None
    workspaces: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class NativeDCPRealization(nn.Module):
    def __init__(
        self,
        backbone: MiniMaxH3DiTModel,
        roles: nn.Embedding,
        *,
        role_embedding: bool,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.role_embeddings = roles
        self.role_embedding_enabled = role_embedding
        self.use_flash_attention = use_flash_attention
        self.options = NativeDCPOptions()
        self.kv_workspaces = [AttentionKVWorkspace() for _ in backbone.blocks]
        self.block_graphs: OrderedDict[DCPBlockGraphKey, CapturedDCPBlocks] = (
            OrderedDict()
        )
        self.graph_captures = 0
        self.graph_replays = 0
        self.fp8_calibration: FP8AttentionCalibration | None = None
        self.fp8_workspaces: list[CalibratedFP8Workspace] = []
        self.fp8_workspace_storage: torch.Tensor | None = None

    @torch.no_grad()
    def prepare_fixed_weights(self, options: NativeDCPOptions) -> None:
        if options.merge_fixed_lora:
            precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            try:
                for name, layer in list(self.named_modules()):
                    if not isinstance(layer, DCPLinear):
                        continue
                    else:
                        pass
                    if layer.frozen is not None:
                        raise ValueError("Fixed adapter merge does not support DeCFG")
                    else:
                        pass
                    adapter = layer.trained
                    if adapter is None:
                        continue
                    else:
                        pass
                    weight = layer.base.weight
                    pairs = (
                        [(adapter.a, adapter.b)]
                        if adapter.a.ndim == 2
                        else zip(adapter.a.unbind(), adapter.b.unbind(), strict=True)
                    )
                    offset = 0
                    for a, b in pairs:
                        for start in range(0, b.shape[0], 1024):
                            stop = min(start + 1024, b.shape[0])
                            target = weight[offset + start : offset + stop]
                            merged = (
                                target.float()
                                + (b[start:stop].float() @ a.float()) * adapter.scale
                            )
                            target.copy_(merged.to(target.dtype))
                        offset += b.shape[0]
                    if layer.reduce_output:
                        layer.base.reduce_results = True
                    else:
                        pass
                    if layer.gather_output:
                        layer.base.gather_output = True
                    else:
                        pass
                    self.set_submodule(name, layer.base)
            finally:
                torch.set_float32_matmul_precision(precision)
        else:
            pass
        if options.fp8_block_linears:
            for block in self.backbone.blocks:
                block.mlp.reuse_fc1_activation = False
                for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                    projection = block.get_submodule(name)
                    layer = (
                        projection.base
                        if isinstance(projection, DCPLinear)
                        else projection
                    )
                    assert isinstance(layer, LinearBase)
                    method = Fp8LinearMethod(Fp8Config())
                    if not method.cutlass_fp8_supported or method.use_marlin:
                        raise ValueError(
                            "Realization FP8 execution requires native CUTLASS FP8"
                        )
                    else:
                        pass
                    method.process_weights_after_loading(layer)
                    layer.quant_method = method
        else:
            pass

    def configure_fp8_attention(self, calibration_path: str) -> None:
        if not self.use_flash_attention or get_ring_ctx()[0] > 1:
            raise ValueError("Calibrated FP8 requires non-Ring FlashAttention")
        else:
            pass
        attention = self.backbone.blocks[0].attn
        self.fp8_calibration = FP8AttentionCalibration.load(
            calibration_path,
            next(self.parameters()).device,
            len(self.backbone.blocks),
            attention.num_heads // get_ulysses_ctx()[0],
        )
        self.clear_execution_cache()

    def configure_execution(self, options: NativeDCPOptions) -> None:
        if options.enable_cuda_graph and (
            not options.reuse_kv_workspace
            or not options.cache_timestep_projections
            or options.cuda_graph_min_history_tokens < 0
            or options.cuda_graph_max_entries < 1
            or options.kv_workspace_capacity_tokens < 0
        ):
            raise ValueError(
                "Realization CUDA Graph requires cached KV/timesteps and valid capture limits"
            )
        else:
            pass
        self.options = options
        for module in self.backbone.modules():
            if isinstance(module, DCPLinear):
                module.fuse_epilogue = options.fuse_adapter_epilogue
                module.reuse_adapter_input_cast = options.reuse_adapter_input_cast
            else:
                pass
        self.clear_execution_cache()

    def clear_execution_cache(self) -> None:
        self.block_graphs.clear()
        self.kv_workspaces = [
            AttentionKVWorkspace(
                self.options.kv_workspace_capacity_tokens,
                batch_history_copies=self.options.batch_history_copies,
            )
            for _ in self.backbone.blocks
        ]
        self.fp8_workspaces = (
            [
                CalibratedFP8Workspace(
                    self.fp8_calibration,
                    index,
                    self.options.kv_workspace_capacity_tokens,
                )
                for index in range(len(self.backbone.blocks))
            ]
            if self.fp8_calibration is not None
            else []
        )
        if (
            self.fp8_calibration is not None
            and self.options.kv_workspace_capacity_tokens
        ):
            shape = (
                len(self.backbone.blocks),
                2,
                1,
                self.options.kv_workspace_capacity_tokens,
                self.fp8_calibration.scales.shape[-1],
                self.fp8_calibration.signs.numel(),
            )
            if (
                self.fp8_workspace_storage is None
                or self.fp8_workspace_storage.shape != shape
                or self.fp8_workspace_storage.device
                != self.fp8_calibration.signs.device
            ):
                self.fp8_workspace_storage = torch.empty(
                    shape,
                    device=self.fp8_calibration.signs.device,
                    dtype=torch.float8_e4m3fn,
                )
            else:
                pass
            for workspace, buffers in zip(
                self.fp8_workspaces, self.fp8_workspace_storage
            ):
                workspace.buffers = (buffers[0], buffers[1])
        else:
            self.fp8_workspace_storage = None

    def timestep_projections(self, times: torch.Tensor) -> TimestepProjections:
        embedding = F.silu(self.backbone.time_embedder(times)).to(torch.bfloat16)
        if get_tp_world_size() > 1:
            local = torch.stack(
                [
                    block.adaln_proj.project_local(embedding)
                    for block in self.backbone.blocks
                ]
            )
            gathered = tensor_model_parallel_all_gather(local)
            blocks = tuple(
                block.adaln_proj.split_output(value)
                for block, value in zip(self.backbone.blocks, gathered)
            )
        else:
            blocks = tuple(
                block.adaln_proj(embedding) for block in self.backbone.blocks
            )
        packed = torch.cat([value.flatten() for block in blocks for value in block])
        offset = 0
        packed_blocks = []
        for block in blocks:
            values = []
            for value in block:
                values.append(packed[offset : offset + value.numel()].view(value.shape))
                offset += value.numel()
            packed_blocks.append(tuple(values))
        return TimestepProjections(
            blocks=tuple(packed_blocks),
            final=self.backbone.final_layer.adaln_proj(embedding),
            packed_blocks=packed,
        )

    def attention(
        self,
        module: MiniMaxH3Attention,
        hidden: torch.Tensor,
        regions: tuple[AttentionRegion, ...],
        *,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        past: LayerKV | None = None,
        retain: torch.Tensor | None = None,
        parallel: bool = False,
        ring: RingPlan | None = None,
        workspace: AttentionKVWorkspace | None = None,
        plan_regions: tuple[AttentionRegion, ...] = (),
        plan_cache: dict[AttentionRegion, torch.Tensor] | None = None,
        fp8_workspace: CalibratedFP8Workspace | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        length = hidden.shape[0]
        projected, _ = module.qkv_proj(hidden)
        q, k, v = (
            item.reshape(length, module.num_heads, module.head_dim)
            for item in projected.split(module.local_inner_dim, dim=-1)
        )
        q, k = _apply_qk_norm(q, k, module.q_norm, module.k_norm, module.head_dim)
        if rope is not None:
            q, k = apply_attention_rope(module, q, k, rope)
        else:
            pass
        degree = get_ulysses_ctx()[0] if parallel else 1
        if degree > 1:
            q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)
        else:
            pass
        k, v = k.unsqueeze(0), v.unsqueeze(0)
        retained = (
            None
            if retain is None
            else (k.index_select(1, retain), v.index_select(1, retain))
        )
        if fp8_workspace is not None:
            fp8_workspace.prepare(past, k.shape[1], k)
            encoded_q, k, v = fp8_workspace.encode(q, k[0], v[0])
        elif past is not None:
            k, v = (
                past.concatenate((k, v))
                if workspace is None
                else workspace.concatenate(past, (k, v))
            )
        else:
            pass
        if ring is not None:
            assert self.use_flash_attention
            result = ring_attention(q, k, v, ring, module.softmax_scale)
        else:
            result = torch.empty_like(q)
            for region in regions:
                if plan_cache is not None and region in plan_cache:
                    result[region.start : region.end] = plan_cache[region][0]
                    continue
                else:
                    pass
                selected_k, selected_v = (
                    (
                        torch.cat(
                            [tensor[:, start:end] for start, end in region.keys], dim=1
                        )
                        if len(region.keys) > 1
                        else tensor[:, region.keys[0][0] : region.keys[0][1]]
                    )
                    for tensor in (k, v)
                )
                query = (q if fp8_workspace is None else encoded_q)[
                    region.start : region.end
                ].unsqueeze(0)
                direct_output = (
                    self.options.direct_attention_output and self.use_flash_attention
                )
                if self.use_flash_attention:
                    output = flash_attn_with_kvcache(
                        query,
                        selected_k,
                        selected_v,
                        softmax_scale=module.softmax_scale,
                        causal=False,
                        q_descale=(
                            None
                            if fp8_workspace is None
                            else fp8_workspace.scales[:, 0]
                        ),
                        k_descale=(
                            None
                            if fp8_workspace is None
                            else fp8_workspace.scales[:, 1]
                        ),
                        v_descale=(
                            None
                            if fp8_workspace is None
                            else fp8_workspace.scales[:, 2]
                        ),
                        num_splits=(
                            self.options.short_query_splits
                            if parallel and query.shape[1] < 64
                            else 1
                        ),
                        out=(
                            result[region.start : region.end].unsqueeze(0)
                            if direct_output and fp8_workspace is None
                            else None
                        ),
                    )
                    if fp8_workspace is not None:
                        output = rotate(
                            output[0],
                            fp8_workspace.signs,
                            center=fp8_workspace.centers[0, 1],
                            inverse=True,
                            out=(
                                result[region.start : region.end]
                                if direct_output
                                else None
                            ),
                        ).unsqueeze(0)
                    else:
                        pass
                else:
                    output = F.scaled_dot_product_attention(
                        query.transpose(1, 2),
                        selected_k.transpose(1, 2),
                        selected_v.transpose(1, 2),
                        scale=module.softmax_scale,
                    ).transpose(1, 2)
                if not direct_output:
                    result[region.start : region.end] = output[0]
                else:
                    pass
                if plan_cache is not None and region in plan_regions:
                    plan_cache[region] = output
                else:
                    pass
        if degree > 1:
            result = _usp_output_all_to_all(result.unsqueeze(0), head_dim=2)
        else:
            pass
        output, _ = module.out_proj(result.reshape(length, module.local_inner_dim))
        return output, retained

    def run_blocks(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        rope_values: torch.Tensor,
        rope_positions: torch.Tensor,
        retained: torch.Tensor | None,
        *,
        regions: tuple[AttentionRegion, ...],
        plan_regions: tuple[AttentionRegion, ...],
        plan_attention_cache: list[dict[AttentionRegion, torch.Tensor]] | None,
        cache: KVCache | None,
        block_adaln: tuple[tuple[torch.Tensor, ...], ...] | None,
        adaln_input: torch.Tensor | None,
        ring: RingPlan | None,
        commit: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        retained_values = []
        for layer_index, block in enumerate(self.backbone.blocks):
            shift, scale, gate, shift_ff, scale_ff, gate_ff = (
                block.adaln_proj(adaln_input)
                if block_adaln is None
                else block_adaln[layer_index]
            )
            normalized = _modulate_rmsnorm_scale_shift(
                hidden, block.norm1, shift, scale, indices, dtype=torch.bfloat16
            )
            past = None if cache is None else cache.layers[layer_index]
            attended, values = self.attention(
                block.attn,
                normalized,
                regions,
                rope=(rope_values, rope_positions),
                past=past,
                retain=retained,
                parallel=True,
                ring=ring,
                plan_regions=plan_regions,
                plan_cache=(
                    plan_attention_cache[layer_index]
                    if plan_attention_cache is not None and ring is None
                    else None
                ),
                workspace=(
                    self.kv_workspaces[layer_index]
                    if self.options.reuse_kv_workspace and ring is None
                    else None
                ),
                fp8_workspace=(
                    self.fp8_workspaces[layer_index] if self.fp8_workspaces else None
                ),
            )
            hidden = _modulate_gate(
                hidden,
                gate,
                attended,
                indices,
                dtype=torch.bfloat16,
                allow_inplace=False,
            )
            normalized = _modulate_rmsnorm_scale_shift(
                hidden, block.norm2, shift_ff, scale_ff, indices, dtype=torch.bfloat16
            )
            hidden = _modulate_gate(
                hidden,
                gate_ff,
                block.mlp(normalized),
                indices,
                dtype=torch.bfloat16,
                allow_inplace=False,
            )
            if commit:
                retained_values.append(values)
            else:
                pass
        return hidden, tuple(retained_values)

    def forward(
        self,
        sequence: RealizationSequence,
        times: torch.Tensor,
        *,
        mode: str = "causal",
        cache: KVCache | None = None,
        commit: bool = False,
        adaln_timestep: str = "old",
        output: str = "prediction",
        timestep_projections: TimestepProjections | None = None,
        plan_attention_cache: list[dict[AttentionRegion, torch.Tensor]] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], KVCache | None]:
        assert mode == "causal" and output == "prediction"
        assert not commit or not torch.is_grad_enabled()
        backbone = self.backbone
        past_length = 0 if cache is None else cache.length
        if self.block_graphs and any(
            workspace.buffers is None
            or workspace.buffers[0].shape[1] < past_length + sequence.length
            for workspace in (self.fp8_workspaces or self.kv_workspaces)
        ):
            self.block_graphs.clear()
        else:
            pass
        key_groups = (() if cache is None else cache.groups) + tuple(
            dict(
                group,
                start=group["start"] + past_length,
                end=group["end"] + past_length,
            )
            for group in sequence.groups
        )
        regions = attention_regions(sequence.groups, key_groups)
        plan_regions = (
            tuple(
                region
                for region in regions
                if any(
                    group["kind"] == "text"
                    and group["start"] <= region.start
                    and region.end <= group["end"]
                    for group in sequence.groups
                )
                and all(
                    group["end"] <= past_length or group["kind"] == "text"
                    for group in key_groups
                    if any(
                        group["start"] < end and start < group["end"]
                        for start, end in region.keys
                    )
                )
            )
            if plan_attention_cache is not None
            else ()
        )
        text, _ = backbone.condition_proj(sequence.refiner_text[0].to(torch.bfloat16))
        if text.shape[0]:
            refiner_regions = attention_regions(
                sequence.refiner_groups, sequence.refiner_groups, refiner=True
            )
            for block in backbone.token_refiner.blocks:
                attended, _ = self.attention(
                    block.attn, block.norm1(text), refiner_regions
                )
                text = text + attended
                text = text + block.mlp(block.norm2(text))
            text = backbone.token_refiner.final_norm(text)
        else:
            pass
        hidden = text.new_zeros((sequence.length, backbone.hidden_size))
        video, _ = backbone.video_patch_proj(sequence.video[0].float())
        audio, _ = backbone.audio_patch_proj(sequence.audio[0].float())
        hidden.index_copy_(0, sequence.text_indices, text[sequence.text_selection])
        hidden.index_copy_(0, sequence.video_indices, video.to(hidden.dtype))
        hidden.index_copy_(0, sequence.audio_indices, audio.to(hidden.dtype))
        if self.role_embedding_enabled:
            hidden = (
                hidden
                + self.role_embeddings(sequence.roles.clamp_min(0)).to(hidden.dtype)
                * (sequence.roles >= 0)[:, None]
            )
        else:
            pass
        adaln_input = (
            F.silu(backbone.time_embedder(times)).to(torch.bfloat16)
            if timestep_projections is None
            else None
        )
        time_ids = build_adaln_timestep_indices(
            sequence, times, mode=mode, adaln_timestep=adaln_timestep
        )
        indices = time_ids * 3 + sequence.tags
        frequencies = backbone.rope(sequence.positions.unsqueeze(0))
        rope_values = _rope_cos_sin_cache(frequencies, dtype=torch.bfloat16)
        rope_positions = torch.arange(sequence.length, device=hidden.device)
        retained, new_groups = (
            commit_rows(sequence.groups, sequence.device, past_length)
            if commit
            else (None, ())
        )
        ring_degree, ring_rank = get_ring_ctx()
        if ring_degree > 1:
            assert get_ulysses_ctx()[0] == 1
            assert cache is None or isinstance(cache, RingKVCache)
            degree, rank = ring_degree, ring_rank
            assert sequence.length % degree == 0
            ring = ring_plan(
                regions,
                sequence.groups,
                sequence.length,
                cache,
                commit=commit,
                device=hidden.device,
            )
            retained = ring.retained
        else:
            degree, rank = get_ulysses_ctx()
            ring = None
        if degree > 1:
            assert sequence.length % degree == 0
            hidden, indices, time_ids, rope_positions = (
                tensor.chunk(degree, dim=0)[rank]
                for tensor in (hidden, indices, time_ids, rope_positions)
            )
        else:
            pass
        if timestep_projections is not None:
            block_adaln = timestep_projections.blocks
        elif get_tp_world_size() > 1:
            # Match native Realization: combine the tiny per-layer AdaLN gathers.
            local_adaln = torch.stack(
                [
                    block.adaln_proj.project_local(adaln_input)
                    for block in backbone.blocks
                ]
            )
            gathered_adaln = tensor_model_parallel_all_gather(local_adaln)
            block_adaln = tuple(
                block.adaln_proj.split_output(value)
                for block, value in zip(backbone.blocks, gathered_adaln)
            )
        else:
            block_adaln = None
        graph_enabled = (
            self.options.enable_cuda_graph
            and cache is not None
            and cache.length >= self.options.cuda_graph_min_history_tokens
            and ring is None
            and timestep_projections is not None
        )
        if graph_enabled:
            assert self.options.reuse_kv_workspace and not torch.is_grad_enabled()
            if self.fp8_workspaces:
                for workspace, history in zip(self.fp8_workspaces, cache.layers):
                    workspace.prepare(history, sequence.length, history.chunks[-1][0])
            else:
                for workspace, history in zip(self.kv_workspaces, cache.layers):
                    workspace.prepare(history, sequence.length)
            inputs = (
                hidden,
                indices,
                rope_values,
                rope_positions,
                timestep_projections.packed_blocks,
            )
            if retained is not None:
                inputs += (retained,)
            else:
                pass
            plan_keys = tuple(tuple(plan) for plan in (plan_attention_cache or ()))
            inputs += tuple(
                plan[region] for plan in (plan_attention_cache or ()) for region in plan
            )
            signature = DCPBlockGraphKey(
                commit=commit,
                history_tokens=cache.length,
                regions=regions,
                plan_regions=plan_regions,
                plan_keys=plan_keys,
                input_shapes=tuple((tensor.shape, tensor.dtype) for tensor in inputs),
                workspace_pointers=tuple(
                    workspace.buffers[0].data_ptr()
                    for workspace in (self.fp8_workspaces or self.kv_workspaces)
                ),
            )
            if (
                self.block_graphs
                and next(iter(self.block_graphs)).workspace_pointers
                != signature.workspace_pointers
            ):
                self.block_graphs.clear()
            else:
                pass
            captured = self.block_graphs.get(signature)
            if captured is None:
                # Bound retained graph memory when sequence layouts change.
                if len(self.block_graphs) == self.options.cuda_graph_max_entries:
                    self.block_graphs.popitem(last=False)
                else:
                    pass
                static_inputs = tuple(tensor.clone() for tensor in inputs)
                plans = (
                    None
                    if plan_attention_cache is None
                    else [{} for _ in backbone.blocks]
                )
                projection_offset = 0
                static_adaln = []
                for block in block_adaln:
                    values = []
                    for value in block:
                        values.append(
                            static_inputs[4][
                                projection_offset : projection_offset + value.numel()
                            ].view(value.shape)
                        )
                        projection_offset += value.numel()
                    static_adaln.append(tuple(values))
                static_adaln = tuple(static_adaln)
                offset = 6 if retained is not None else 5
                for plan, keys in zip(plans or (), plan_keys):
                    for region in keys:
                        plan[region] = static_inputs[offset]
                        offset += 1
                # Register captured all-reduce buffers with the TP peers.
                with get_tp_group().graph_capture() as capture_context:
                    stream = capture_context.stream
                    for _ in range(2):
                        self.run_blocks(
                            *static_inputs[:4],
                            static_inputs[5] if retained is not None else None,
                            regions=regions,
                            plan_regions=plan_regions,
                            plan_attention_cache=(
                                [dict(plan) for plan in plans]
                                if plans is not None
                                else None
                            ),
                            cache=cache,
                            block_adaln=static_adaln,
                            adaln_input=adaln_input,
                            ring=None,
                            commit=commit,
                        )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        graph_hidden, graph_retained = self.run_blocks(
                            *static_inputs[:4],
                            static_inputs[5] if retained is not None else None,
                            regions=regions,
                            plan_regions=plan_regions,
                            plan_attention_cache=plans,
                            cache=cache,
                            block_adaln=static_adaln,
                            adaln_input=adaln_input,
                            ring=None,
                            commit=commit,
                        )
                torch.cuda.current_stream().wait_stream(stream)
                captured = CapturedDCPBlocks(
                    graph=graph,
                    inputs=static_inputs,
                    hidden=graph_hidden,
                    retained=graph_retained,
                    plans=plans,
                    workspaces=tuple(
                        workspace.buffers
                        for workspace in (self.fp8_workspaces or self.kv_workspaces)
                    ),
                )
                self.block_graphs[signature] = captured
                self.graph_captures += 1
            else:
                self.block_graphs.move_to_end(signature)
                for target, source in zip(captured.inputs, inputs):
                    target.copy_(source)
            captured.graph.replay()
            self.graph_replays += 1
            hidden = captured.hidden
            # Captured outputs are scratch; committed chunks must survive replay.
            retained_values = tuple(
                tuple(tensor.clone() for tensor in values)
                for values in captured.retained
            )
            for plan, saved in zip(plan_attention_cache or (), captured.plans or ()):
                plan.update(saved)
        else:
            hidden, retained_values = self.run_blocks(
                hidden,
                indices,
                rope_values,
                rope_positions,
                retained,
                regions=regions,
                plan_regions=plan_regions,
                plan_attention_cache=plan_attention_cache,
                cache=cache,
                block_adaln=block_adaln,
                adaln_input=adaln_input,
                ring=ring,
                commit=commit,
            )
        layers = [
            (ChunkedLayerKV() if cache is None else cache.layers[index]).append(values)
            for index, values in enumerate(retained_values)
        ]
        shift, scale = (
            backbone.final_layer.adaln_proj(adaln_input)
            if timestep_projections is None
            else timestep_projections.final
        )
        hidden = _modulate_rmsnorm_scale_shift(
            hidden,
            backbone.final_layer.norm,
            shift,
            scale,
            time_ids,
            dtype=torch.bfloat16,
        )
        if degree > 1:
            hidden = get_sp_group().all_gather(hidden, dim=0)
        else:
            pass
        video, _ = backbone.final_layer.video_out(
            hidden[sequence.video_indices].float()
        )
        audio, _ = backbone.final_layer.audio_out(
            hidden[sequence.audio_indices].float()
        )
        if get_tp_world_size() > 1:
            video = tensor_model_parallel_all_gather(video)
            audio = tensor_model_parallel_all_gather(audio)
        else:
            pass
        if commit:
            groups = (() if cache is None else cache.groups) + new_groups
            if ring is not None:
                new_cache = RingKVCache(tuple(layers), groups, ring.shards, rank)
            else:
                new_cache = KVCache(tuple(layers), groups)
        else:
            new_cache = cache
        return (video.unsqueeze(0), audio.unsqueeze(0)), new_cache


class CachedTimestepModel(nn.Module):
    """Cache immutable inference weights' projections for one session's sampler."""

    def __init__(self, model: NativeDCPRealization, schedule: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.denoise_step = 0
        self.projections: list[TimestepProjections] = []
        self.clean_projection: TimestepProjections | None = None
        self.plan_attention_cache: list[dict[AttentionRegion, torch.Tensor]] | None = (
            None
        )

    def forward(
        self,
        sequence: RealizationSequence,
        times: torch.Tensor,
        *,
        cache: KVCache | None = None,
        commit: bool = False,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], KVCache | None]:
        if not self.projections:
            assert commit and cache is None
            self.clean_projection = self.model.timestep_projections(times)
            for timestep in self.schedule:
                scheduled_times = times.clone()
                scheduled_times[1:] = timestep
                self.projections.append(
                    self.model.timestep_projections(scheduled_times)
                )
        else:
            pass
        if commit:
            projection = self.clean_projection
            self.denoise_step = 0
            self.plan_attention_cache = None
        else:
            if self.denoise_step == 0 and self.model.options.reuse_plan_attention:
                self.plan_attention_cache = [{} for _ in self.model.backbone.blocks]
            else:
                pass
            projection = self.projections[self.denoise_step]
            self.denoise_step += 1
        return self.model(
            sequence,
            times,
            cache=cache,
            commit=commit,
            timestep_projections=projection,
            plan_attention_cache=self.plan_attention_cache,
        )


@torch.no_grad()
def native_dcp_from_reference(
    reference: RealizationTransformer,
    *,
    device: torch.device,
    use_flash_attention: bool = True,
) -> NativeDCPRealization:
    """Transfer every restored DCP parameter; retain runtime adapters unmerged."""
    reference.cpu()
    config = MiniMaxH3DiTConfig()
    config.update_model_arch(dict(reference.config))
    config.arch_config.adaln_out_features = 18 * reference.config.hidden_size
    config.arch_config.final_adaln_out_features = 2 * reference.config.hidden_size
    with torch.device("meta"):
        backbone = MiniMaxH3DiTModel(config, hf_config={})
    state = (
        (name.replace(".base_layer.", "."), value)
        for name, value in reference.state_dict().items()
        if ".lora_" not in name
        and ".decfg." not in name
        and name != "role_embeddings.weight"
    )
    converted: dict[str, torch.Tensor] = {}
    for name, value in _diffusers_h3_checkpoint(state):
        if get_tp_world_size() > 1:
            # Convert and transfer one local shard at a time; retaining a full
            # converted CPU copy on every rank doubles host-memory pressure.
            module_name, parameter_name = name.rsplit(".", 1)
            layer = backbone.get_submodule(module_name)
            if isinstance(layer, ColumnParallelLinear):
                if parameter_name == "bias":
                    value = value.chunk(layer.tp_size)[layer.tp_rank].contiguous()
                else:
                    value = shard_column(value, layer)
            elif isinstance(layer, RowParallelLinear) and parameter_name == "weight":
                value = value.chunk(layer.tp_size, dim=1)[layer.tp_rank].contiguous()
            else:
                pass
            value = value.to(device)
        else:
            pass
        converted[name] = value
    converted["rope.inv_freq"] = reference.rope.inv_freq.to(device)
    backbone.load_state_dict(converted, strict=True, assign=True)
    backbone.post_load_weights()
    mapping = get_param_names_mapping(backbone.param_names_mapping)
    linears: dict[str, list[tuple[int, nn.Module]]] = defaultdict(list)
    for name, module in reference.named_modules():
        if (
            isinstance(module, (nn.Linear, PeftLinear))
            and ".base_layer" not in name
            and ".lora_" not in name
            and ".decfg." not in name
        ):
            target, index, _ = mapping(name + ".weight")
            linears[target.removesuffix(".weight")].append(
                (0 if index is None else index, module)
            )
        else:
            pass
    for name, entries in linears.items():
        sources = [module for _, module in sorted(entries, key=lambda pair: pair[0])]
        frozen = adapter_delta(sources, frozen=True, swap_mlp=name.endswith(".mlp.fc1"))
        trained = adapter_delta(
            sources, frozen=False, swap_mlp=name.endswith(".mlp.fc1")
        )
        layer = backbone.get_submodule(name)
        if get_tp_world_size() > 1:
            shard_adapter(frozen, layer)
            shard_adapter(trained, layer)
        else:
            pass
        if frozen is not None or trained is not None:
            backbone.set_submodule(name, DCPLinear(layer, frozen, trained))
        else:
            pass
    model = NativeDCPRealization(
        backbone,
        reference.role_embeddings,
        role_embedding=reference.role_embedding_enabled,
        use_flash_attention=use_flash_attention,
    )
    return model.to(device).eval().requires_grad_(False)
