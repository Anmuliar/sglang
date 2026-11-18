# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Centered FP8 QKV transforms and inverse Hadamard kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def transform(value, signs, dim: tl.constexpr, block: tl.constexpr):
    d = tl.arange(0, dim)
    value *= tl.load(signs + d)[None, :]
    for step in tl.static_range(7):
        bit = 1 << step
        partner = tl.gather(
            value, tl.broadcast_to((d ^ bit)[None, :], (block, dim)), axis=1
        )
        value = tl.where((d[None, :] & bit) == 0, value + partner, partner - value)
    return (value * dim**-0.5).to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["rows", "layer"])
def encode_qkv(
    q,
    k,
    v,
    qo,
    q_route,
    ko,
    vo,
    past,
    centers,
    scales,
    signs,
    clips,
    rows,
    heads: tl.constexpr,
    dim: tl.constexpr,
    layer,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    audit: tl.constexpr,
    block: tl.constexpr,
    route_query: tl.constexpr,
    text_capacity: tl.constexpr = 0,
    text_valid_rows=None,
):
    r = tl.program_id(0) * block + tl.arange(0, block)
    h, side = tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, dim)
    if tl.program_id(2) == 0:
        value = tl.load(
            q + r[:, None] * qs0 + h * qs1 + d[None, :], (r < rows)[:, None], 0
        )
    elif tl.program_id(2) == 1:
        value = tl.load(
            k + r[:, None] * ks0 + h * ks1 + d[None, :], (r < rows)[:, None], 0
        )
    else:
        value = tl.load(
            v + r[:, None] * vs0 + h * vs1 + d[None, :], (r < rows)[:, None], 0
        )
    value = value.to(tl.float32)
    if tl.program_id(2) > 0:
        center = tl.load(centers + ((layer * 2 + side - 1) * heads + h) * dim + d)
        value -= center[None, :]
    else:
        pass
    value = transform(value, signs, dim, block)
    scale = tl.load(scales + (layer * 3 + side) * heads + h)
    normalized = value / scale
    if audit:
        count = tl.sum(
            tl.sum(((tl.abs(normalized) > 448) & (r < rows)[:, None]).to(tl.int32), 1),
            0,
        )
        tl.atomic_add(clips + (layer * 3 + side) * heads + h, count)
    else:
        pass
    value = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
    if tl.program_id(2) == 0:
        encoded = value.to(tl.float8e4nv)
        tl.store(
            qo + (r[:, None] * heads + h) * dim + d[None, :],
            encoded,
            (r < rows)[:, None],
        )
        if route_query:
            tl.store(
                q_route + (r[:, None] * heads + h) * dim + d[None, :],
                encoded.to(tl.float32) * scale,
                (r < rows)[:, None],
            )
        else:
            pass
    else:
        offset = tl.load(past).to(tl.int64)
        target = r
        active = r < rows
        if text_capacity:
            real = tl.load(text_valid_rows)
            target = tl.where(r < text_capacity, r, r - text_capacity + real)
            active &= (r < real) | (r >= text_capacity)
        else:
            pass
        if tl.program_id(2) == 1:
            tl.store(
                ko + ((target[:, None] + offset) * heads + h) * dim + d[None, :],
                value,
                active[:, None],
            )
        else:
            tl.store(
                vo + ((target[:, None] + offset) * heads + h) * dim + d[None, :],
                value,
                active[:, None],
            )


@triton.jit
def butterfly(
    x,
    signs,
    center,
    count,
    output,
    rows: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    output_stride0: tl.constexpr,
    output_stride1: tl.constexpr,
    inverse: tl.constexpr,
    centered: tl.constexpr,
    bounded: tl.constexpr,
    block_rows: tl.constexpr,
):
    r = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    head = tl.program_id(1)
    d = tl.arange(0, dim)
    valid = r < rows
    if bounded:
        valid = valid & (r < tl.load(count))
    else:
        pass
    value = tl.load(
        x + r[:, None] * stride0 + head * stride1 + d[None, :], valid[:, None], 0
    ).to(tl.float32)
    sign = tl.load(signs + d)
    if not inverse:
        if centered:
            mean = tl.load(center + head * dim + d)
            value = tl.where(valid[:, None], value - mean[None, :], 0.0)
        else:
            pass
        value *= sign[None, :]
    else:
        pass
    for step in tl.static_range(7):
        bit = 1 << step
        indices = tl.broadcast_to((d ^ bit)[None, :], (block_rows, dim))
        partner = tl.gather(value, indices, axis=1)
        value = tl.where((d[None, :] & bit) == 0, value + partner, partner - value)
    value *= dim**-0.5
    if inverse:
        value *= sign[None, :]
        if centered:
            value += tl.load(center + head * dim + d)[None, :]
        else:
            pass
    else:
        pass
    tl.store(
        output + r[:, None] * output_stride0 + head * output_stride1 + d[None, :],
        value,
        (r < rows)[:, None],
    )


def rotate(
    x: torch.Tensor,
    signs: torch.Tensor,
    *,
    center: torch.Tensor | None = None,
    count: torch.Tensor | None = None,
    inverse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    rows, heads, dim = x.shape
    output = (
        torch.empty_like(x, memory_format=torch.contiguous_format)
        if out is None
        else out
    )
    butterfly[(triton.cdiv(rows, 4), heads)](
        x,
        signs,
        center,
        count,
        output,
        rows,
        heads,
        dim,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        inverse,
        center is not None,
        count is not None,
        4,
        num_warps=4,
    )
    return output
