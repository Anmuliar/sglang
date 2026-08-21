# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.llada_image import LLaDAImageSamplingParams
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.llada_image import (
    LLaDAImageTransformerBlock,
    _LLaDAImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning import (
    format_llada_image_prompt,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import (
    get_global_server_args,
    set_global_server_args,
)


class _CaptureSPBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_values = []
        self.replicated_suffixes = []

    def forward(self, hidden_states, *args, **kwargs):
        self.skip_values.append(kwargs.get("skip_sequence_parallel_override", False))
        self.replicated_suffixes.append(kwargs.get("num_replicated_suffix", 0))
        return hidden_states


class TestLLaDAImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.previous_server_args = get_global_server_args()
        except ValueError:
            cls.previous_server_args = None
        set_global_server_args(SimpleNamespace())

    @classmethod
    def tearDownClass(cls):
        set_global_server_args(cls.previous_server_args)

    def test_dit_supports_sglang_flash_attention_and_sdpa(self):
        backends = (
            LLaDAImagePipelineConfig().dit_config.arch_config._supported_attention_backends
        )
        self.assertEqual(
            backends,
            {AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

    def test_dit_weight_mapping_fuses_qkv_and_swiglu_inputs(self):
        mapping = get_param_names_mapping(
            LLaDAImagePipelineConfig().dit_config.arch_config.param_names_mapping
        )

        self.assertEqual(
            mapping("layers.0.attention.to_k.weight"),
            ("layers.0.attention.to_qkv.weight", 1, 3),
        )
        self.assertEqual(
            mapping("layers.0.feed_forward.w1.weight"),
            ("layers.0.feed_forward.w13.weight", 0, 2),
        )
        self.assertEqual(
            mapping("layers.0.feed_forward.w3.weight"),
            ("layers.0.feed_forward.w13.weight", 1, 2),
        )

    def test_prompt_format_matches_official_pipeline(self):
        self.assertEqual(
            format_llada_image_prompt("a red car"),
            "<role>HUMAN</role> Generate an image: a red car\n"
            "<role>ASSISTANT</role>\n<IMAGE1>",
        )
        self.assertEqual(
            format_llada_image_prompt(None),
            "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>",
        )

    def test_pipeline_config_uses_llada_image_schedule_and_shape(self):
        config = LLaDAImagePipelineConfig()
        sigmas = config.prepare_sigmas(None, num_inference_steps=8)

        self.assertEqual(len(sigmas), 8)
        self.assertTrue(all(left > right for left, right in pairwise(sigmas)))
        self.assertEqual(
            config.prepare_latent_shape(
                SimpleNamespace(height=1024, width=768),
                batch_size=1,
                num_frames=1,
            ),
            (1, 128, 64, 48),
        )

    def test_pipeline_config_rejects_unaligned_resolution(self):
        config = LLaDAImagePipelineConfig()
        with self.assertRaisesRegex(ValueError, "height must be divisible by 16"):
            config.prepare_latent_shape(
                SimpleNamespace(height=1023, width=1024),
                batch_size=1,
                num_frames=1,
            )

    def test_service_sampling_params_do_not_expose_vq_mode(self):
        field_names = {field.name for field in fields(LLaDAImageSamplingParams)}

        self.assertNotIn("generation_mode", field_names)
        self.assertNotIn("vq_token_ids", field_names)

    def test_dit_block_forwards_sequence_parallel_controls(self):
        class CaptureAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received = None

            def forward(
                self,
                hidden_states,
                attention_mask,
                freqs_cis,
                num_replicated_suffix=0,
                skip_sequence_parallel_override=False,
            ):
                self.received = (
                    num_replicated_suffix,
                    skip_sequence_parallel_override,
                )
                return torch.zeros_like(hidden_states)

        block = object.__new__(LLaDAImageTransformerBlock)
        torch.nn.Module.__init__(block)
        block.modulation = False
        block.attention = CaptureAttention()
        block.attention_norm1 = torch.nn.Identity()
        block.attention_norm2 = torch.nn.Identity()
        block.ffn_norm1 = torch.nn.Identity()
        block.ffn_norm2 = torch.nn.Identity()
        block.feed_forward = torch.nn.Identity()

        block(
            torch.ones(1, 4, 8),
            attention_mask=None,
            freqs_cis=torch.empty(0),
            num_replicated_suffix=32,
            skip_sequence_parallel_override=True,
        )

        self.assertEqual(block.attention.received, (32, True))

    def test_edit_sp_shards_images_and_replicates_conditions(self):
        model_module = "sglang.multimodal_gen.runtime.models.dits.llada_image"
        linear_module = "sglang.multimodal_gen.runtime.layers.linear"
        with (
            patch(f"{model_module}.get_tp_world_size", return_value=1),
            patch(f"{linear_module}.get_tp_group", return_value=None),
            patch(f"{linear_module}.get_group_size", return_value=1),
            patch(f"{linear_module}.get_group_rank", return_value=0),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
                return_value=SimpleNamespace(attention_backend="torch_sdpa"),
            ),
        ):
            model = _LLaDAImageTransformer2DModel(
                in_channels=4,
                dim=64,
                n_layers=1,
                n_refiner_layers=1,
                n_heads=2,
                cap_feat_dim=8,
                semantic_feat_dim=10,
                axes_dims=(8, 12, 12),
                axes_lens=(256, 32, 32),
            )

        noise_refiner = _CaptureSPBlock()
        context_refiner = _CaptureSPBlock()
        sigvq_refiner = _CaptureSPBlock()
        main_block = _CaptureSPBlock()
        model.noise_refiner = torch.nn.ModuleList([noise_refiner])
        model.context_refiner = torch.nn.ModuleList([context_refiner])
        model.sigvq_refiner = torch.nn.ModuleList([sigvq_refiner])
        model.layers = torch.nn.ModuleList([main_block])

        with (
            patch(f"{model_module}.get_sp_world_size", return_value=2),
            patch(f"{model_module}.get_sp_parallel_rank", return_value=0),
            torch.no_grad(),
        ):
            model(
                x=[torch.randn(4, 1, 4, 4)],
                t=torch.tensor([0.5]),
                cap_feats=[torch.randn(3, 8)],
                glm_cap_feats=[torch.randn(5, 10)],
                source_latents=[torch.randn(4, 1, 4, 4)],
            )

        self.assertEqual(noise_refiner.skip_values, [False])
        self.assertEqual(context_refiner.skip_values, [True])
        self.assertEqual(sigvq_refiner.skip_values, [True])
        self.assertEqual(main_block.skip_values, [False])
        self.assertEqual(main_block.replicated_suffixes, [96])

    def test_edit_skips_sigvq_refiner_for_empty_cfg_condition(self):
        model_module = "sglang.multimodal_gen.runtime.models.dits.llada_image"
        linear_module = "sglang.multimodal_gen.runtime.layers.linear"
        with (
            patch(f"{model_module}.get_tp_world_size", return_value=1),
            patch(f"{linear_module}.get_tp_group", return_value=None),
            patch(f"{linear_module}.get_group_size", return_value=1),
            patch(f"{linear_module}.get_group_rank", return_value=0),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
                return_value=SimpleNamespace(attention_backend="torch_sdpa"),
            ),
        ):
            model = _LLaDAImageTransformer2DModel(
                in_channels=4,
                dim=64,
                n_layers=1,
                n_refiner_layers=1,
                n_heads=2,
                cap_feat_dim=8,
                semantic_feat_dim=10,
                axes_dims=(8, 12, 12),
                axes_lens=(256, 32, 32),
            )

        noise_refiner = _CaptureSPBlock()
        context_refiner = _CaptureSPBlock()
        sigvq_refiner = _CaptureSPBlock()
        main_block = _CaptureSPBlock()
        model.noise_refiner = torch.nn.ModuleList([noise_refiner])
        model.context_refiner = torch.nn.ModuleList([context_refiner])
        model.sigvq_refiner = torch.nn.ModuleList([sigvq_refiner])
        model.layers = torch.nn.ModuleList([main_block])

        with (
            patch(f"{model_module}.get_sp_world_size", return_value=1),
            patch(f"{model_module}.get_sp_parallel_rank", return_value=0),
            torch.no_grad(),
        ):
            model(
                x=[torch.randn(4, 1, 4, 4)],
                t=torch.tensor([0.5]),
                cap_feats=[torch.randn(3, 8)],
                glm_cap_feats=[torch.empty(0, 10)],
                source_latents=[torch.randn(4, 1, 4, 4)],
            )

        self.assertEqual(noise_refiner.skip_values, [False])
        self.assertEqual(context_refiner.skip_values, [True])
        self.assertEqual(sigvq_refiner.skip_values, [])
        self.assertEqual(main_block.skip_values, [False])

    def test_sp_uses_global_height_coordinates_for_generation_and_edit(self):
        calls = []

        def pad_with_ids(
            features, grid_size, start, noise_value=None, sequence_multiple=32
        ):
            del sequence_multiple
            length = len(features)
            calls.append((grid_size, start))
            return (
                features,
                torch.zeros((length, 3), dtype=torch.int32),
                torch.zeros(length, dtype=torch.bool),
                length,
                [noise_value] * length if noise_value is not None else None,
            )

        subject = SimpleNamespace(
            _patchify_image=lambda _image, _patch, _f_patch: (
                torch.zeros(8, 1),
                (1, 2, 4),
                (1, 2, 4),
            ),
            _pad_with_ids=pad_with_ids,
        )
        module = "sglang.multimodal_gen.runtime.models.dits.llada_image"
        with (
            patch(f"{module}.get_sp_world_size", return_value=2),
            patch(f"{module}.get_sp_parallel_rank", return_value=1),
        ):
            _LLaDAImageTransformer2DModel._prepare_t2i_sequences(
                subject,
                [torch.zeros(1, 1, 2, 4)],
                cap_feats=None,
                glm_features=None,
                patch_size=1,
                f_patch_size=1,
            )

        self.assertEqual(calls[-1], ((1, 2, 4), (1, 2, 0)))

        calls.clear()
        with (
            patch(f"{module}.get_sp_world_size", return_value=2),
            patch(f"{module}.get_sp_parallel_rank", return_value=1),
        ):
            _LLaDAImageTransformer2DModel._prepare_editing_sequences(
                subject,
                [torch.zeros(1, 1, 2, 4)],
                cap_feats=[torch.zeros(1, 1)],
                glm_cap_feats=[torch.zeros(1, 1)],
                source_latents=[torch.zeros(1, 1, 2, 4)],
                patch_size=1,
                f_patch_size=1,
            )

        image_calls = [call for call in calls if call[0] == (1, 2, 4)]
        self.assertEqual(len(image_calls), 2)
        self.assertTrue(all(start[1:] == (2, 0) for _, start in image_calls))
        self.assertEqual(calls[-1], ((1, 1, 1), (35, 0, 0)))


if __name__ == "__main__":
    unittest.main()
