"""Load released per-channel FP8 Realization weights without requantization."""

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from sgl_kernel import sgl_per_token_quant_fp8
from torch import nn

from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_dcp import (
    NativeDCPRealization,
    shard_column,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MINIMAX_H3_FP32_BUFFER_NAMES,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3DiTModel,
    _reorder_grouped_qkv_to_qkv,
)

logger = logging.getLogger(__name__)


class ModelOptFp8DirectMethod(Fp8LinearMethod):
    """Run the fixed per-channel format through existing SGLang kernels."""

    def apply(
        self, layer: nn.Module, value: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        matrix = value.contiguous().view(-1, value.shape[-1])
        quantized = torch.empty_like(matrix, dtype=torch.float8_e4m3fn)
        scales = torch.empty(
            (matrix.shape[0], 1), device=value.device, dtype=torch.float32
        )
        sgl_per_token_quant_fp8(matrix, quantized, scales)
        output = fp8_scaled_mm(
            quantized,
            layer.weight,
            scales,
            layer.weight_scale,
            out_dtype=value.dtype,
            bias=bias,
        )
        return output.view(*value.shape[:-1], output.shape[-1])


def modelopt_tensor(
    name: str, value: torch.Tensor, layer: nn.Module, *, heads: int, head_dim: int
) -> torch.Tensor:
    """Apply the same row permutation and TP ownership to weights and scales."""
    scale = name.endswith(".weight_scale")
    if ".attn.qkv_proj." in name:
        value = _reorder_grouped_qkv_to_qkv(
            value, num_query_groups=heads, heads_per_group=1, head_dim=head_dim
        )
    else:
        pass
    if isinstance(layer, ColumnParallelLinear):
        if name.endswith(".bias"):
            value = value.chunk(layer.tp_size, dim=0)[layer.tp_rank].contiguous()
        elif scale:
            value = shard_column(value[:, None], layer)
            value = value[:, 0]
        else:
            value = shard_column(value, layer)
    elif isinstance(layer, RowParallelLinear) and name.endswith(".weight"):
        value = value.chunk(layer.tp_size, dim=1)[layer.tp_rank].contiguous()
    else:
        pass
    return value


@torch.no_grad()
def native_dcp_from_modelopt(
    folder: str, *, device: torch.device, direct_kernels: bool = False
) -> NativeDCPRealization:
    root = Path(folder)
    manifest = json.loads((root / "complete.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if (
        manifest["format"] != "duo-release-modelopt-v1"
        or config["quantization_config"]["quant_algo"] != "FP8_PER_CHANNEL_PER_TOKEN"
    ):
        raise ValueError("Unsupported Duo ModelOpt checkpoint layout")
    else:
        pass
    realization = manifest["models"]["realization"]
    layout = config.get("h3_checkpoint_layout")
    if layout is not None:
        if (
            layout["qkv"] != "original_per_head_interleaved"
            or layout["fc1"] != "gate_then_value"
            or not layout["merged_lora"]
        ):
            raise ValueError("Unsupported Duo ModelOpt checkpoint layout")
        else:
            role_embedding = layout["role_embedding_enabled"]
    elif (
        config["_class_name"] == "MiniMaxH3DiTModel"
        and realization["mode"] == "full"
        and realization["lora"] is None
    ):
        role_embedding = realization["role_embedding"]
    else:
        raise ValueError("Unsupported Duo ModelOpt checkpoint layout")
    architecture = realization["config"]
    native_config = MiniMaxH3DiTConfig()
    native_config.update_model_arch(architecture)
    native_config.arch_config.adaln_out_features = config["adaln_out_features"]
    native_config.arch_config.final_adaln_out_features = config[
        "final_adaln_out_features"
    ]
    with torch.device("meta"):
        backbone = MiniMaxH3DiTModel(native_config, hf_config={})
    expected = backbone.state_dict()
    index = json.loads(
        (root / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    restored: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    roles: torch.Tensor | None = None
    for shard in sorted(set(index.values())):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                value = handle.get_tensor(name)
                if name == "role_embeddings.weight":
                    roles = value.to(device)
                    continue
                else:
                    pass
                module_name, parameter_name = name.rsplit(".", 1)
                layer = backbone.get_submodule(module_name)
                value = modelopt_tensor(
                    name,
                    value,
                    layer,
                    heads=config["num_attention_heads"],
                    head_dim=config["attention_head_dim"],
                )
                if parameter_name == "weight_scale":
                    scales[module_name] = value.to(
                        device=device, dtype=torch.float32
                    ).reshape(1, -1)
                    continue
                else:
                    pass
                if tuple(value.shape) != tuple(expected[name].shape):
                    raise ValueError(f"ModelOpt TP shape mismatch: {name}")
                elif (
                    name in MINIMAX_H3_FP32_PARAM_NAMES
                    or name in MINIMAX_H3_FP32_BUFFER_NAMES
                ):
                    value = value.float()
                else:
                    pass
                restored[name] = value.to(device)
    backbone.load_state_dict(restored, strict=True, assign=True)
    backbone.post_load_weights()
    for module_name, scale in scales.items():
        layer = backbone.get_submodule(module_name)
        if (
            layer.weight.dtype != torch.float8_e4m3fn
            or scale.numel() != layer.weight.shape[0]
        ):
            raise ValueError(f"Invalid per-channel FP8 weight: {module_name}")
        else:
            pass
        method_class = ModelOptFp8DirectMethod if direct_kernels else Fp8LinearMethod
        method = method_class(
            Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic")
        )
        if not method.cutlass_fp8_supported or method.use_marlin:
            raise ValueError(
                "ModelOpt Realization requires the native FP8 GEMM backend"
            )
        else:
            pass
        layer.weight = nn.Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = nn.Parameter(scale, requires_grad=False)
        layer.input_scale = None
        layer.quant_method = method
    if roles is None:
        raise ValueError("ModelOpt checkpoint is missing role embeddings")
    else:
        pass
    for block in backbone.blocks:
        block.mlp.reuse_fc1_activation = False
    model = NativeDCPRealization(
        backbone,
        nn.Embedding.from_pretrained(roles, freeze=True),
        role_embedding=role_embedding,
    )
    logger.info(
        f"Loaded ModelOpt FP8 Realization: path={root}, quantized_linears={len(scales)}, parameter_bytes={sum(p.numel() * p.element_size() for p in model.parameters())}"
    )
    return model.eval().requires_grad_(False)
