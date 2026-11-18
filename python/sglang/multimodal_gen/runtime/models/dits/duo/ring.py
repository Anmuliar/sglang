"""Sequence-sharded clean KV and group-masked Realization ring attention."""

from dataclasses import dataclass

import torch
import torch.distributed as distributed
from torchtitan.models.duo_realization.modules.kv_cache import KVCache

from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_with_kvcache
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_sp_group,
)
from sglang.multimodal_gen.runtime.layers.usp import _ring_merge_attention


@dataclass(frozen=True)
class AttentionRegion:
    start: int
    end: int
    keys: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RingKVCache(KVCache):
    """Global group coordinates with local, full-head LayerKV storage."""

    shards: tuple[tuple[tuple[int, int], ...], ...]
    rank: int

    def evict(self, first_block: int) -> "RingKVCache":
        kept = tuple(
            group
            for group in self.groups
            if group["kind"] != "clean" or group["block"] >= first_block
        )
        if len(kept) == len(self.groups):
            return self
        else:
            pass
        groups, cursor = [], 0
        for group in kept:
            length = group["end"] - group["start"]
            groups.append(dict(group, start=cursor, end=cursor + length))
            cursor += length
        shards, local_ranges = [], ()
        for rank, shard in enumerate(self.shards):
            ranges, remapped, offset = [], [], 0
            for start, end in shard:
                for old, new in zip(kept, groups):
                    lo, hi = max(start, old["start"]), min(end, old["end"])
                    if lo < hi:
                        ranges.append((offset + lo - start, offset + hi - start))
                        remapped.append(
                            (
                                new["start"] + lo - old["start"],
                                new["start"] + hi - old["start"],
                            )
                        )
                    else:
                        pass
                offset += end - start
            shards.append(tuple(remapped))
            if rank == self.rank:
                local_ranges = tuple(ranges)
            else:
                pass
        return RingKVCache(
            tuple(layer.select_ranges(local_ranges) for layer in self.layers),
            tuple(groups),
            tuple(shards),
            self.rank,
        )


@dataclass(frozen=True)
class RingPlan:
    regions: tuple[AttentionRegion, ...]
    keys: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    max_length: int
    retained: torch.Tensor | None
    shards: tuple[tuple[tuple[int, int], ...], ...]


def ring_plan(
    regions: tuple[AttentionRegion, ...],
    groups: tuple[dict[str, str | int], ...],
    length: int,
    cache: RingKVCache | None,
    *,
    commit: bool,
    device: torch.device,
) -> RingPlan:
    degree, rank = get_ring_ctx()
    chunk = length // degree
    past_length = 0 if cache is None else cache.length
    local_regions = tuple(
        AttentionRegion(
            max(region.start, rank * chunk) - rank * chunk,
            min(region.end, (rank + 1) * chunk) - rank * chunk,
            region.keys,
        )
        for region in regions
        if region.start < (rank + 1) * chunk and region.end > rank * chunk
    )
    keys, shards, lengths, retained = [], [], [], None
    for owner in range(degree):
        old_shard = () if cache is None else cache.shards[owner]
        current_start = owner * chunk
        current_end = current_start + chunk
        physical = old_shard + (
            (past_length + current_start, past_length + current_end),
        )
        owner_keys = []
        for region in local_regions:
            selected, offset = [], 0
            for start, end in physical:
                for key_start, key_end in region.keys:
                    lo, hi = max(start, key_start), min(end, key_end)
                    if lo < hi:
                        interval = (offset + lo - start, offset + hi - start)
                        if selected and selected[-1][1] == interval[0]:
                            selected[-1] = (selected[-1][0], interval[1])
                        else:
                            selected.append(interval)
                    else:
                        pass
                offset += end - start
            owner_keys.append(tuple(selected))
        keys.append(tuple(owner_keys))
        lengths.append(sum(end - start for start, end in physical))
        additions, rows, cursor = [], [], past_length
        if commit:
            for group in groups:
                if group["kind"] in ("reference", "shared", "clean"):
                    lo = max(current_start, group["start"])
                    hi = min(current_end, group["end"])
                    if lo < hi:
                        additions.append(
                            (cursor + lo - group["start"], cursor + hi - group["start"])
                        )
                        rows.extend(range(lo - current_start, hi - current_start))
                    else:
                        pass
                    cursor += group["end"] - group["start"]
                else:
                    pass
            if owner == rank:
                retained = torch.tensor(rows, dtype=torch.long, device=device)
            else:
                pass
        else:
            pass
        shards.append(old_shard + tuple(additions))
    return RingPlan(local_regions, tuple(keys), max(lengths), retained, tuple(shards))


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: RingPlan,
    scale: float,
) -> torch.Tensor:
    """Rotate bounded KV buffers; merge visible partial softmaxes in FP32."""
    degree, rank = get_ring_ctx()
    group = get_sp_group().ring_group
    next_rank = distributed.get_global_rank(group, (rank + 1) % degree)
    previous_rank = distributed.get_global_rank(group, (rank - 1) % degree)
    # Padding is transport-only. The CPU range plan never exposes it to FA3.
    buffers = [
        k.new_empty((2, 1, plan.max_length, k.shape[2], k.shape[3])) for _ in range(2)
    ]
    buffers[0][0, :, : k.shape[1]].copy_(k)
    buffers[0][1, :, : v.shape[1]].copy_(v)
    outputs: list[torch.Tensor | None] = [None] * len(plan.regions)
    normalizers: list[torch.Tensor | None] = [None] * len(plan.regions)
    for step in range(degree):
        current, following = step % 2, (step + 1) % 2
        if step < degree - 1:
            pending = distributed.batch_isend_irecv(
                [
                    distributed.P2POp(
                        distributed.isend, buffers[current], next_rank, group
                    ),
                    distributed.P2POp(
                        distributed.irecv, buffers[following], previous_rank, group
                    ),
                ]
            )
        else:
            pending = []
        owner = (rank - step) % degree
        for index, (region, ranges) in enumerate(zip(plan.regions, plan.keys[owner])):
            if not ranges:
                continue
            else:
                pass
            key, value = (
                (
                    torch.cat([tensor[:, start:end] for start, end in ranges], dim=1)
                    if len(ranges) > 1
                    else tensor[:, ranges[0][0] : ranges[0][1]]
                )
                for tensor in buffers[current]
            )
            output, lse, *_ = flash_attn_with_kvcache(
                q[region.start : region.end].unsqueeze(0),
                key,
                value,
                softmax_scale=scale,
                causal=False,
                num_splits=1,
                return_softmax_lse=True,
            )
            outputs[index], normalizers[index] = _ring_merge_attention(
                outputs[index], normalizers[index], output, lse
            )
        for operation in pending:
            operation.wait()
    result = torch.empty_like(q)
    for region, output in zip(plan.regions, outputs):
        assert output is not None
        result[region.start : region.end] = output[0].to(q.dtype)
    return result
