# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)


class TestLLaDAImagePipelineConfig(unittest.TestCase):
    def setUp(self):
        self.config = LLaDAImagePipelineConfig()

    def test_edit_keeps_requested_default_output_size(self):
        image = Image.new("RGB", (768, 512))

        self.assertIsNone(
            self.config.calculate_condition_image_size(image, image.width, image.height)
        )
        self.assertIsNone(self.config.prepare_calculated_size(image))

    def test_decode_preprocessing_matches_official_vae_dtype_order(self):
        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.bfloat16), requires_grad=False
                )
                self.bn = SimpleNamespace(
                    running_mean=torch.tensor(
                        [0.01, -0.02, 0.03, -0.04], dtype=torch.float32
                    ),
                    running_var=torch.full((4,), 0.0129973, dtype=torch.float32),
                )
                self.config = SimpleNamespace(
                    arch_config=SimpleNamespace(batch_norm_eps=0.003)
                )

        vae = FakeVAE()
        latents = torch.tensor(
            [[[[0.501]], [[-0.249]], [[0.126]], [[-0.751]]]],
            dtype=torch.float32,
        )
        official_latents = latents.to(torch.bfloat16)
        latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(official_latents)
        latent_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.arch_config.batch_norm_eps
        ).to(official_latents)
        expected = official_latents * latent_std + latent_mean
        expected = expected.reshape(1, 1, 2, 2, 1, 1)
        expected = expected.permute(0, 1, 4, 2, 5, 3).reshape(1, 1, 2, 2)

        actual = self.config.preprocess_decoding(latents, vae=vae)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_edit_sp_shards_target_and_source_latents(self):
        latents = torch.randn(1, 128, 4, 4)
        source_latents = torch.randn(128, 1, 4, 4)
        batch = SimpleNamespace(
            batch_size=1,
            condition_image=Image.new("RGB", (64, 64)),
            image_embeds=None,
            source_latents=[source_latents],
            enable_sequence_shard=False,
        )
        base_module = "sglang.multimodal_gen.configs.pipeline_configs.base"
        llada_module = "sglang.multimodal_gen.configs.pipeline_configs.llada_image"
        with (
            patch(f"{base_module}.get_sp_world_size", return_value=2),
            patch(f"{base_module}.get_sp_parallel_rank", return_value=0),
            patch(f"{llada_module}.get_sp_world_size", return_value=2),
        ):
            actual, did_shard = self.config.shard_latents_for_sp(batch, latents)
            batch.did_sp_shard_latents = did_shard
            source = self.config.prepare_pos_cond_kwargs(
                batch, torch.device("cpu"), rotary_emb=None, dtype=torch.float32
            )["source_latents"][0]

        self.assertEqual(actual.shape, (1, 128, 2, 4))
        torch.testing.assert_close(actual, latents[:, :, :2, :])
        self.assertTrue(did_shard)
        self.assertEqual(source.shape, (128, 1, 2, 4))
        torch.testing.assert_close(source, source_latents[:, :, :2, :])

    def test_generation_sp_rejects_latent_height_padding(self):
        latents = torch.randn(1, 128, 63, 64)
        batch = SimpleNamespace(
            condition_image=None,
            source_latents=None,
            enable_sequence_shard=False,
        )
        module = "sglang.multimodal_gen.configs.pipeline_configs.llada_image"
        with (
            patch(f"{module}.get_sp_world_size", return_value=2),
            self.assertRaisesRegex(
                ValueError, "latent height 63 must be divisible by SP degree 2"
            ),
        ):
            self.config.shard_latents_for_sp(batch, latents)

    def test_validates_supported_parallel_topology(self):
        defaults = dict(
            sp_degree=2,
            ulysses_degree=2,
            ring_degree=1,
            tp_size=1,
            dp_size=1,
            cfg_parallel_degree=1,
            num_gpus=2,
        )
        self.config.validate_server_args(SimpleNamespace(**defaults))

        invalid_cases = [
            ({"ulysses_degree": 1, "ring_degree": 2}, "ring_degree=1"),
            (
                {"sp_degree": 4, "ulysses_degree": 4, "num_gpus": 4},
                "supports only SP degrees 1 and 2",
            ),
            ({"cfg_parallel_degree": 2}, "TP=DP=CFG=1"),
            ({"sp_degree": 1, "ulysses_degree": 1}, "num_gpus == sp_degree"),
        ]
        for overrides, message in invalid_cases:
            args = defaults | overrides
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, message
            ):
                self.config.validate_server_args(SimpleNamespace(**args))


if __name__ == "__main__":
    unittest.main()
