# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.multimodal_gen.configs.models.dits.llada_image import LLaDAImageDitConfig
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size


@dataclass
class LLaDAImagePipelineConfig(SpatialImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2I
    should_use_guidance: bool = False

    dit_config: LLaDAImageDitConfig = field(default_factory=LLaDAImageDitConfig)
    dit_precision: str = "bf16"
    vae_config: Flux2VAEConfig = field(default_factory=Flux2VAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple = ()
    text_encoder_precisions: tuple[str, ...] = ()
    preprocess_text_funcs: tuple = ()
    postprocess_text_funcs: tuple = ()

    latent_scale_factor: int = 16
    text_encoder_mem_fraction_static: float = 0.35

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is not None:
            return sigmas
        schedule = np.linspace(0.001, 1.0, num_inference_steps + 1)[:-1]
        schedule = (1 - (1 - schedule**1.17) ** 0.8) ** 1.1
        return (1 - schedule).tolist()

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        del num_frames
        if batch.height % self.latent_scale_factor != 0:
            raise ValueError("LLaDA-Image height must be divisible by 16")
        if batch.width % self.latent_scale_factor != 0:
            raise ValueError("LLaDA-Image width must be divisible by 16")
        return (
            batch_size,
            self.dit_config.num_channels_latents,
            batch.height // self.latent_scale_factor,
            batch.width // self.latent_scale_factor,
        )

    def prepare_calculated_size(self, image):
        del image
        return None

    def calculate_condition_image_size(self, image, width, height):
        del image, width, height
        return None

    def shard_latents_for_sp(self, batch, latents):
        sp_degree = get_sp_world_size()
        if latents.dim() == 4 and latents.shape[2] % sp_degree != 0:
            raise ValueError(
                f"LLaDA-Image latent height {latents.shape[2]} must be divisible "
                f"by SP degree {sp_degree}; choose a compatible output height"
            )
        return super().shard_latents_for_sp(batch, latents)

    def validate_server_args(self, server_args) -> None:
        super().validate_server_args(server_args)
        # Replicated condition suffixes are currently de-duplicated only by the
        # Ulysses attention path.
        if server_args.ring_degree != 1:
            raise ValueError("LLaDA-Image sequence parallelism requires ring_degree=1")
        if server_args.ulysses_degree != server_args.sp_degree:
            raise ValueError(
                "LLaDA-Image sequence parallelism requires "
                "ulysses_degree == sp_degree"
            )
        if server_args.sp_degree not in (1, 2):
            raise ValueError("LLaDA-Image currently supports only SP degrees 1 and 2")
        if (
            server_args.tp_size != 1
            or server_args.dp_size != 1
            or server_args.cfg_parallel_degree != 1
        ):
            raise ValueError("LLaDA-Image requires diffusion parallelism TP=DP=CFG=1")
        if server_args.num_gpus != server_args.sp_degree:
            raise ValueError(
                "LLaDA-Image requires num_gpus == sp_degree so every GPU belongs "
                "to the sequence-parallel group"
            )

    def validate_num_outputs_per_prompt(
        self, num_outputs_per_prompt: int, server_args
    ) -> None:
        if server_args.sp_degree > 1 and num_outputs_per_prompt != 1:
            raise ValueError(
                "LLaDA-Image sequence parallelism supports only n=1; "
                "submit separate requests for multiple images"
            )

    @staticmethod
    def _prepare_condition_list(values, name: str, expected_size: int, device, dtype):
        if values is None:
            return None
        if len(values) != expected_size:
            raise ValueError(
                f"LLaDA-Image {name} has {len(values)} entries, "
                f"expected {expected_size}"
            )
        return [value.to(device=device, dtype=dtype) for value in values]

    def _prepare_source_latents(self, batch, device, dtype):
        source_latents = self._prepare_condition_list(
            batch.source_latents,
            "source_latents",
            batch.batch_size,
            device,
            dtype,
        )
        if source_latents and getattr(batch, "did_sp_shard_latents", False):
            source_latents = [
                self.shard_latents_for_sp(batch, latent)[0] for latent in source_latents
            ]
        return source_latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del rotary_emb
        image_embeds = batch.image_embeds
        if image_embeds:
            image_embeds = self._prepare_condition_list(
                image_embeds,
                "image_embeds",
                batch.batch_size,
                device,
                dtype,
            )
        source_latents = self._prepare_source_latents(batch, device, dtype)
        return {
            "encoder_hidden_states_image": image_embeds,
            "source_latents": source_latents,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del rotary_emb
        image_embeds = batch.image_embeds
        if image_embeds:
            image_embeds = self._prepare_condition_list(
                image_embeds,
                "image_embeds",
                batch.batch_size,
                device,
                dtype,
            )
            empty_image_embed = image_embeds[0].new_zeros(
                (0, image_embeds[0].shape[-1])
            )
            image_embeds = [empty_image_embed] * batch.batch_size
        source_latents = self._prepare_source_latents(batch, device, dtype)
        return {
            "encoder_hidden_states_image": image_embeds,
            "source_latents": source_latents,
        }

    def get_decode_scale_and_shift(self, device, dtype, vae):
        del device, dtype, vae
        return 1.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        del server_args
        if vae is None or not hasattr(vae, "bn"):
            raise ValueError("LLaDA-Image decoding requires the Flux2 VAE BN state")
        vae_parameter = next(vae.parameters())
        latents = latents.to(device=vae_parameter.device, dtype=vae_parameter.dtype)
        vae_config = getattr(vae.config, "arch_config", vae.config)
        latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
        latent_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae_config.batch_norm_eps
        ).to(latents)
        latents = latents * latent_std + latent_mean
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    def post_denoising_loop(self, latents, batch):
        del batch
        return latents
