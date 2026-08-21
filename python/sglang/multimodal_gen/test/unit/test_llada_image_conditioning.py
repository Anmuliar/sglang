# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning import (
    LLaDAImageTextConditioningStage,
    LLaDAImageTextEncoderRunner,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class _FakeTextRunner:
    def __init__(self):
        self.prompts = None

    def encode(self, prompts, max_sequence_length):
        self.prompts = (prompts, max_sequence_length)
        return [
            torch.full((index + 2, 4), float(index + 1))
            for index in range(len(prompts))
        ]


class TestLLaDAImageTextConditioning(unittest.TestCase):
    def setUp(self):
        self.runner = _FakeTextRunner()
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            self.stage = LLaDAImageTextConditioningStage(self.runner)

    def test_repeats_positive_and_negative_text_for_each_output(self):
        batch = SimpleNamespace(
            prompt="a red car",
            negative_prompt=None,
            guidance_scale=5.0,
            num_outputs_per_prompt=2,
            max_sequence_length=128,
        )

        result = self.stage.forward(batch, server_args=SimpleNamespace())

        self.assertTrue(result.do_classifier_free_guidance)
        self.assertEqual(len(result.prompt_embeds), 2)
        self.assertEqual(len(result.negative_prompt_embeds), 2)
        self.assertEqual(len(result.prompt_attention_mask), 2)
        self.assertEqual(len(result.negative_attention_mask), 2)
        self.assertTrue(torch.equal(result.prompt_embeds[0], result.prompt_embeds[1]))
        self.assertTrue(
            torch.equal(
                result.negative_prompt_embeds[0], result.negative_prompt_embeds[1]
            )
        )

    def test_guidance_disabled_has_no_negative_condition(self):
        batch = SimpleNamespace(
            prompt="a red car",
            negative_prompt=None,
            guidance_scale=1.0,
            num_outputs_per_prompt=2,
            max_sequence_length=128,
        )

        result = self.stage.forward(batch, server_args=SimpleNamespace())

        self.assertFalse(result.do_classifier_free_guidance)
        self.assertEqual(len(result.prompt_embeds), 2)
        self.assertEqual(result.negative_prompt_embeds, [])
        self.assertEqual(result.negative_attention_mask, [])

    def test_text_runner_uses_sp_group_as_text_encoder_tp_group(self):
        fake_worker = SimpleNamespace(
            model_runner=object(),
            model_config=object(),
            get_memory_pool=lambda: (object(), object()),
        )
        srt_args_module = "sglang.srt.server_args.ServerArgs"
        worker_module = "sglang.srt.managers.tp_worker.TpModelWorker"
        with (
            patch(
                worker_module,
                return_value=fake_worker,
            ) as worker_cls,
            patch(
                "sglang.srt.mem_cache.cache_init_params.CacheInitParams",
                return_value=object(),
            ),
            patch(
                "sglang.srt.mem_cache.chunk_cache.ChunkCache",
                return_value=object(),
            ),
            patch(
                srt_args_module,
                side_effect=lambda **kwargs: SimpleNamespace(page_size=1, **kwargs),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning.get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning.get_sp_parallel_rank",
                return_value=1,
            ),
        ):
            runner = LLaDAImageTextEncoderRunner(
                model_root="/unused/model",
                queryformer=object(),
                text_projection=object(),
                tokenizer=object(),
                server_args=SimpleNamespace(
                    sp_degree=2,
                    nccl_port=29500,
                    pipeline_config=SimpleNamespace(
                        text_encoder_mem_fraction_static=0.1
                    ),
                ),
            )

        self.assertIs(runner.worker, fake_worker)
        self.assertEqual(runner.server_args.tp_size, 2)
        self.assertEqual(worker_cls.call_args.kwargs["tp_rank"], 1)


class TestLLaDAImageConditionKwargs(unittest.TestCase):
    def setUp(self):
        self.config = LLaDAImagePipelineConfig()
        self.semantic = [
            torch.full((3, 5), 1.0),
            torch.full((3, 5), 2.0),
        ]
        self.source = [
            torch.full((8, 1, 2, 3), 3.0),
            torch.full((8, 1, 2, 3), 4.0),
        ]
        self.batch = SimpleNamespace(
            batch_size=2,
            image_embeds=self.semantic,
            source_latents=self.source,
            do_classifier_free_guidance=True,
        )

    def test_cfg_uses_semantics_only_on_positive_branch(self):
        positive = self.config.prepare_pos_cond_kwargs(
            self.batch, torch.device("cpu"), rotary_emb=None, dtype=torch.float64
        )
        negative = self.config.prepare_neg_cond_kwargs(
            self.batch, torch.device("cpu"), rotary_emb=None, dtype=torch.float64
        )
        policy = CFGPolicy().build(
            self.batch,
            {"encoder_hidden_states_image": self.semantic},
            positive,
            negative,
        )

        positive_kwargs = policy.branches[0].kwargs
        negative_kwargs = policy.branches[1].kwargs
        self.assertEqual(
            [tuple(x.shape) for x in positive_kwargs["encoder_hidden_states_image"]],
            [(3, 5)] * 2,
        )
        self.assertEqual(
            [tuple(x.shape) for x in negative_kwargs["encoder_hidden_states_image"]],
            [(0, 5)] * 2,
        )
        self.assertTrue(
            all(x.dtype == torch.float64 for x in positive_kwargs["source_latents"])
        )
        for positive_source, negative_source in zip(
            positive_kwargs["source_latents"],
            negative_kwargs["source_latents"],
            strict=True,
        ):
            torch.testing.assert_close(positive_source, negative_source)

    def test_rejects_condition_batch_length_mismatch(self):
        self.batch.image_embeds = self.semantic[:1]

        with self.assertRaisesRegex(
            ValueError, "image_embeds has 1 entries, expected 2"
        ):
            self.config.prepare_pos_cond_kwargs(
                self.batch,
                torch.device("cpu"),
                rotary_emb=None,
                dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
