# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibrated FP8 attention workspaces for immutable causal Realization history."""

from dataclasses import dataclass

import torch
import triton
from torchtitan.models.duo_realization.modules.kv_cache import LayerKV

from sglang.multimodal_gen.runtime.models.dits.duo.fp8_kernels import encode_qkv


class SM90FP8Attention:
    """Use the optional full-API FA3 build without importing another engine."""

    local_heads: int = 7
    minimum_query_tokens: int = 65

    def __init__(self, library_path: str) -> None:
        torch.ops.load_library(library_path)
        self.metadata: dict[
            tuple[int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scales: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        rows, heads, dimension = q.shape
        capacity = k.shape[1]
        shape = (rows, capacity, heads, dimension)
        if shape not in self.metadata:
            query_offsets = torch.tensor([0, rows], device=q.device, dtype=torch.int32)
            key_offsets = torch.tensor(
                [0, capacity], device=q.device, dtype=torch.int32
            )
            lengths = key_offsets[1:]
            scheduler = torch.ops.h3_stock_fa3_async_n128.get_scheduler_metadata(
                1,
                rows,
                capacity,
                heads,
                heads,
                dimension,
                dimension,
                torch.float8_e4m3fn,
                lengths,
                query_offsets,
                None,
                None,
                None,
                None,
                None,
                0,
                False,
                -1,
                -1,
                False,
                1,
                None,
                0,
            )
            self.metadata[shape] = query_offsets, key_offsets, scheduler
        else:
            pass
        query_offsets, key_offsets, scheduler = self.metadata[shape]
        output = torch.empty_like(q, dtype=torch.bfloat16)
        return torch.ops.h3_stock_fa3_async_n128.fwd(
            q,
            k[0],
            v[0],
            None,
            None,
            None,
            output,
            query_offsets,
            key_offsets,
            None,
            None,
            None,
            rows,
            capacity,
            None,
            None,
            None,
            None,
            None,
            None,
            scales[:, 0],
            scales[:, 1],
            scales[:, 2],
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            True,
            scheduler,
            1,
            None,
            0,
            None,
            1,
            0,
            None,
        )[0]


@dataclass(kw_only=True)
class FP8AttentionCalibration:
    centers: torch.Tensor
    scales: torch.Tensor
    signs: torch.Tensor

    @classmethod
    def load(
        cls, path: str, device: torch.device, layers: int, heads: int
    ) -> "FP8AttentionCalibration":
        profile = torch.load(path, map_location=device, weights_only=True)
        centers, scales, signs = (
            profile[name] for name in ("centers", "scales", "signs")
        )
        if (
            profile["schema"] != "h3_rotated_static_v1"
            or centers.shape != (layers, 2, heads, 128)
            or scales.shape != (layers, 3, heads)
            or signs.shape != (128,)
            or not torch.isfinite(centers).all()
            or not (torch.isfinite(scales) & (scales > 0)).all()
            or not (signs.abs() == 1).all()
        ):
            raise ValueError(
                "Calibration does not match the Realization attention partition"
            )
        else:
            return cls(
                centers=centers.float(), scales=scales.float(), signs=signs.float()
            )


class CalibratedFP8Workspace:
    def __init__(
        self, calibration: FP8AttentionCalibration, layer: int, capacity_tokens: int
    ) -> None:
        self.centers = calibration.centers[layer : layer + 1]
        self.scales = calibration.scales[layer : layer + 1]
        self.signs = calibration.signs
        self.capacity_tokens = capacity_tokens
        self.buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        self.history: LayerKV | None = None
        self.encoded_chunks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.past_tokens = torch.zeros((), device=self.signs.device, dtype=torch.int32)
        self.history_tokens = 0
        self.encoded_history_tokens = 0

    def encode(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, heads, dimension = q.shape
        output_q = torch.empty_like(
            q, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format
        )
        key_buffer, value_buffer = self.buffers
        encode_qkv[(triton.cdiv(rows, 4), heads, 3)](
            q,
            k,
            v,
            output_q,
            None,
            key_buffer,
            value_buffer,
            self.past_tokens,
            self.centers,
            self.scales,
            self.signs,
            None,
            rows,
            heads,
            dimension,
            0,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            False,
            4,
            False,
            num_warps=4,
        )
        stop = self.history_tokens + rows
        return output_q, key_buffer[:, :stop], value_buffer[:, :stop]

    def prepare(
        self, history: LayerKV | None, current_tokens: int, template: torch.Tensor
    ) -> None:
        chunks = () if history is None else history.chunks
        history_tokens = sum(chunk[0].shape[1] for chunk in chunks)
        capacity = max(self.capacity_tokens, history_tokens + current_tokens)
        if self.buffers is None or self.buffers[0].shape[1] < capacity:
            self.buffers = tuple(
                torch.empty(
                    (1, capacity, *template.shape[2:]),
                    device=template.device,
                    dtype=torch.float8_e4m3fn,
                )
                for _ in range(2)
            )
            self.history = None
        else:
            pass
        if self.history is history and self.history_tokens == history_tokens:
            return
        else:
            pass

        # Retain only live chunk encodings; object identity prevents stale KV reuse.
        previous = self.encoded_chunks
        previous_offsets: dict[int, int] = {}
        offset = 0
        if self.history is not None:
            for chunk in self.history.chunks:
                previous_offsets[id(chunk[0])] = offset
                offset += chunk[0].shape[1]
        else:
            pass
        live: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        sources: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        offset = 0
        for chunk in chunks:
            identity = id(chunk[0])
            length = chunk[0].shape[1]
            if identity in previous:
                encoded = previous[identity]
                if previous_offsets.get(identity) != offset:
                    for buffer, tensor in zip(self.buffers, encoded):
                        destinations.append(buffer[:, offset : offset + length])
                        sources.append(tensor)
                else:
                    pass
            else:
                self.history_tokens = offset
                self.past_tokens.fill_(offset)
                _, keys, values = self.encode(chunk[0][0], chunk[0][0], chunk[1][0])
                encoded = (
                    keys[:, offset : offset + length].clone(),
                    values[:, offset : offset + length].clone(),
                )
                self.encoded_history_tokens += length
            live[identity] = encoded
            offset += length
        if sources:
            torch._foreach_copy_(destinations, sources)  # PyTorch batched copy API.
        else:
            pass
        self.encoded_chunks = live
        self.history = history
        self.history_tokens = history_tokens
        self.past_tokens.fill_(history_tokens)
