"""Load and shard a Duo Realization engine independently of the serving frontend."""

import json
from argparse import Namespace
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
from torchtitan.components.inferencer_realization import (
    RealizationInferencer,
    load_model,
)

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.duo.kernels import (
    use_realization_kernels,
)
from sglang.multimodal_gen.runtime.models.dits.duo.modelopt import (
    native_dcp_from_modelopt,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_dcp import (
    native_dcp_from_reference,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    NativeDCPOptions,
)


def load_realization_engine(
    args: Namespace,
    *,
    device: torch.device,
    use_sglang_kernels: bool,
    native_backbone: bool,
    rendezvous: str,
    sp_mode: Literal["ulysses", "ring"],
    tp_size: int,
    native_options: NativeDCPOptions | None,
) -> tuple[Namespace, RealizationInferencer]:
    resolved_args = copy(args)
    manifest = json.loads((Path(args.checkpoint) / "manifest.json").read_text())
    resolved_args.origin_intent = manifest["identity"]["qwen"].get("mode") == "origin"
    engine = RealizationInferencer(resolved_args, device=device)
    if native_backbone:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=tp_size,
            sp_size=args.context_parallel_degree // tp_size,
            ulysses_degree=(
                args.context_parallel_degree // tp_size if sp_mode == "ulysses" else 1
            ),
            ring_degree=(
                args.context_parallel_degree // tp_size if sp_mode == "ring" else 1
            ),
            distributed_init_method=rendezvous,
        )
        if (
            native_options is not None
            and native_options.modelopt_checkpoint is not None
        ):
            engine.model = native_dcp_from_modelopt(
                native_options.modelopt_checkpoint,
                device=device,
                direct_kernels=native_options.fp8_direct_kernels,
            )
        elif tp_size > 1:
            # Restore on CPU before sharding so startup does not require a full
            # Realization replica on each GPU.
            engine.model, _, _ = load_model(
                resolved_args,
                SimpleNamespace(cp=args.context_parallel_degree),
                torch.device("cpu"),
            )
            engine.model = native_dcp_from_reference(engine.model, device=device)
        else:
            engine.ensure_realization()
            engine.model = native_dcp_from_reference(engine.model, device=device)
        if native_options is not None:
            engine.model.configure_execution(native_options)
            if native_options.merge_fixed_lora or native_options.fp8_block_linears:
                engine.model.prepare_fixed_weights(native_options)
            else:
                pass
            if native_options.fp8_attention_calibration is not None:
                engine.model.configure_fp8_attention(
                    native_options.fp8_attention_calibration.format(
                        rank=torch.distributed.get_rank()
                    )
                )
            else:
                pass
        else:
            pass
        engine.residency.register("realization", engine.model)
        engine.residency.move("realization", device, "native")
        torch.cuda.empty_cache()
    else:
        engine.ensure_realization()
        if use_sglang_kernels:
            use_realization_kernels(engine.model)
        else:
            pass
    return resolved_args, engine
