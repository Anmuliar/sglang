"""Inference workspaces and rounding-preserving adapter epilogues."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torchtitan.models.duo_realization.modules.kv_cache import LayerKV


@dataclass(kw_only=True)
class NativeDCPOptions:
    modelopt_checkpoint: str | None = None
    fp8_direct_kernels: bool = False
    reuse_kv_workspace: bool = True
    fuse_adapter_epilogue: bool = True
    cache_timestep_projections: bool = True
    short_query_splits: int = 1
    reuse_plan_attention: bool = True
    enable_cuda_graph: bool = False
    cuda_graph_min_history_tokens: int = 0
    cuda_graph_max_entries: int = 8
    kv_workspace_capacity_tokens: int = 0
    batch_history_copies: bool = False
    reuse_adapter_input_cast: bool = False
    defer_finite_checks: bool = False
    direct_attention_output: bool = False
    merge_fixed_lora: bool = False
    fp8_block_linears: bool = False
    fp8_attention_calibration: str | None = None


class ChunkedLayerKV(LayerKV):
    """Select immutable chunk intervals without enumerating token indices."""

    def append(self, values: tuple[torch.Tensor, torch.Tensor]) -> "ChunkedLayerKV":
        return ChunkedLayerKV(
            self.chunks + (tuple(tensor.detach() for tensor in values),)
        )

    def select_ranges(
        self, ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...]
    ) -> "ChunkedLayerKV":
        selected = []
        offset = 0
        for chunk in self.chunks:
            length = chunk[0].shape[1]
            spans = [
                (max(start, offset) - offset, min(stop, offset + length) - offset)
                for start, stop in ranges
                if max(start, offset) < min(stop, offset + length)
            ]
            if spans and sum(stop - start for start, stop in spans) == length:
                selected.append(chunk)
            elif spans:
                # Copy partial chunks so eviction can release their original storage.
                selected.append(
                    tuple(
                        torch.cat(
                            [tensor[:, start:stop] for start, stop in spans], dim=1
                        )
                        for tensor in chunk
                    )
                )
            else:
                pass
            offset += length
        if not selected and self.chunks:
            selected.append(tuple(tensor[:, :0].clone() for tensor in self.chunks[0]))
        else:
            pass
        return ChunkedLayerKV(tuple(selected))


class AttentionKVWorkspace:
    def __init__(
        self, capacity_tokens: int = 0, *, batch_history_copies: bool = False
    ) -> None:
        self.capacity_tokens = capacity_tokens
        self.batch_history_copies = batch_history_copies
        self.history: LayerKV | None = None
        self.buffers: tuple[torch.Tensor, torch.Tensor] | None = None

    def prepare(self, history: LayerKV, current_tokens: int) -> None:
        prefix_tokens = sum(chunk[0].shape[1] for chunk in history.chunks)
        total_tokens = max(self.capacity_tokens, prefix_tokens + current_tokens)
        template = history.chunks[-1]
        if (
            self.buffers is None
            or self.buffers[0].shape[1] < total_tokens
            or self.buffers[0].shape[2:] != template[0].shape[2:]
            or self.buffers[0].dtype != template[0].dtype
            or self.buffers[0].device != template[0].device
        ):
            self.buffers = tuple(
                tensor.new_empty((tensor.shape[0], total_tokens, *tensor.shape[2:]))
                for tensor in template
            )
            self.history = None
        else:
            pass
        if self.history is not history:
            previous_chunks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            if self.batch_history_copies and self.history is not None:
                offset = 0
                for chunk in self.history.chunks:
                    previous_chunks[offset] = chunk
                    offset += chunk[0].shape[1]
            else:
                pass
            destinations: list[torch.Tensor] = []
            sources: list[torch.Tensor] = []
            offset = 0
            for chunk in history.chunks:
                stop = offset + chunk[0].shape[1]
                previous = previous_chunks.get(offset)
                if previous is None or not all(
                    old is new for old, new in zip(previous, chunk)
                ):
                    for buffer, tensor in zip(self.buffers, chunk):
                        if self.batch_history_copies:
                            destinations.append(buffer[:, offset:stop])
                            sources.append(tensor)
                        else:
                            buffer[:, offset:stop].copy_(tensor)
                else:
                    pass
                offset = stop
            if destinations:
                torch._foreach_copy_(destinations, sources)  # PyTorch batched copy API.
            else:
                pass
            self.history = history
        else:
            pass

    def concatenate(
        self, history: LayerKV, current: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not history.chunks:
            return current
        else:
            pass
        self.prepare(history, current[0].shape[1])
        prefix_tokens = sum(chunk[0].shape[1] for chunk in history.chunks)
        total_tokens = prefix_tokens + current[0].shape[1]
        for buffer, tensor in zip(self.buffers, current):
            buffer[:, prefix_tokens:total_tokens].copy_(tensor)
        return tuple(buffer[:, :total_tokens] for buffer in self.buffers)


@triton.jit
def adapter_epilogue_kernel(
    base_pointer,
    frozen_pointer,
    trained_pointer,
    output_pointer,
    elements: tl.constexpr,
    frozen_scale: tl.constexpr,
    trained_scale: tl.constexpr,
    has_frozen: tl.constexpr,
    has_trained: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < elements
    dtype = base_pointer.dtype.element_ty
    result = tl.load(base_pointer + offsets, mask, other=0).to(tl.float32)
    if has_frozen:
        frozen = tl.load(frozen_pointer + offsets, mask, other=0) * frozen_scale
        result = (result + frozen.to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    else:
        pass
    if has_trained:
        trained = tl.load(trained_pointer + offsets, mask, other=0) * trained_scale
        result = result + trained
    else:
        pass
    tl.store(output_pointer + offsets, result.to(dtype), mask)


def adapter_epilogue(
    base: torch.Tensor,
    frozen: torch.Tensor | None,
    trained: torch.Tensor | None,
    frozen_scale: float,
    trained_scale: float,
) -> torch.Tensor:
    assert base.is_contiguous()
    assert frozen is None or frozen.is_contiguous()
    assert trained is None or trained.is_contiguous()
    output = torch.empty_like(base)
    block_size = 1024
    adapter_epilogue_kernel[(triton.cdiv(base.numel(), block_size),)](
        base,
        frozen,
        trained,
        output,
        base.numel(),
        frozen_scale,
        trained_scale,
        frozen is not None,
        trained is not None,
        block_size,
        enable_fp_fusion=False,
    )
    return output
