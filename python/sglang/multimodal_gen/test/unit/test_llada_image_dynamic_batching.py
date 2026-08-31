# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)


def _request(prompt: str, seed: int, image_path=None) -> Req:
    return Req(
        sampling_params=SamplingParams(
            data_type=DataType.IMAGE,
            prompt=prompt,
            image_path=image_path,
            seed=seed,
            guidance_scale=1.0,
            num_inference_steps=4,
            num_outputs_per_prompt=1,
        ),
        extra={},
    )


def _scheduler(pipeline_config) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=pipeline_config)
    return scheduler


def test_llada_image_merges_text_to_image_requests():
    scheduler = _scheduler(LLaDAImagePipelineConfig())
    scheduler._batch_admission = SimpleNamespace(enabled=True)
    requests = [_request("a red car", 11), _request("a blue boat", 22)]

    assert scheduler._dynamic_batching_enabled()
    merged = scheduler._try_merge_generation_reqs(requests)

    assert merged is not None
    assert merged.prompt == ["a red car", "a blue boat"]
    assert merged.image_path is None
    assert merged.extra["dynamic_batch_seeds"] == [11, 22]


def test_llada_image_merges_edit_requests_and_preserves_source_order():
    scheduler = _scheduler(LLaDAImagePipelineConfig())
    requests = [
        _request("make it sunny", 11, "/images/red.png"),
        _request("make it rainy", 22, ["/images/blue.png"]),
    ]

    merged = scheduler._try_merge_generation_reqs(requests)

    assert merged is not None
    assert merged.prompt == ["make it sunny", "make it rainy"]
    assert merged.image_path == ["/images/red.png", "/images/blue.png"]
    assert merged.extra["dynamic_batch_seeds"] == [11, 22]


def test_llada_image_does_not_mix_text_to_image_and_edit_requests():
    scheduler = _scheduler(LLaDAImagePipelineConfig())

    assert (
        scheduler._try_merge_generation_reqs(
            [
                _request("a red car", 11),
                _request("make it rainy", 22, "/images/blue.png"),
            ]
        )
        is None
    )


def test_image_batching_remains_opt_in_for_other_pipelines():
    scheduler = _scheduler(PipelineConfig())
    first = _request("make it sunny", 11, "/images/red.png")
    second = _request("make it rainy", 22, "/images/blue.png")

    assert not scheduler._can_dynamic_batch(first, second)


def test_multi_source_request_is_not_dynamic_batch_eligible():
    scheduler = _scheduler(LLaDAImagePipelineConfig())
    request = _request(
        "combine the images",
        11,
        ["/images/red.png", "/images/blue.png"],
    )

    assert not scheduler._can_dynamic_batch(request, request)


def test_dynamic_batch_seeds_expand_in_request_major_order():
    batch = SimpleNamespace(
        seed=11,
        prompt=["a red car", "a blue boat"],
        num_outputs_per_prompt=2,
        extra={"dynamic_batch_seeds": [11, 22]},
        generator_device="cpu",
    )
    stage = object.__new__(InputValidationStage)

    stage._generate_seeds(
        batch,
        SimpleNamespace(
            pipeline_config=SimpleNamespace(generator_device=None),
        ),
    )

    assert batch.seeds == [11, 12, 22, 23]
    assert len(batch.generator) == 4
