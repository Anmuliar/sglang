"""Native Realization VAE loading, reference encoding, and bounded temporal decoding."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.duo import DuoDecodeMode
from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_group_coordinator,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.loader.utils import (
    load_model_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.duo.solarx import TileStreamingDecoder
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import (
    MiniMaxH3AudioVAE,
    MiniMaxH3VideoVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention as VideoAttention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    DiagonalGaussianDistribution,
    get_tile_parallel_state,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class DuoCodecs:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        parallel_audio_decode: bool = False,
        prepare_video_weights: bool = False,
        parallel_video_decode: bool = False,
        video_decode_backend: Literal["native", "solarx"] = "native",
        video_use_graph: bool = True,
        load_audio: bool = True,
        video_attention_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
    ) -> None:
        root = Path(model_path)
        video_settings = json.loads((root / "FL2VA/video_vae/config.json").read_text())
        audio_settings = json.loads((root / "audio_vae/config.json").read_text())
        video_config = MiniMaxH3VideoVAEConfig(
            use_parallel_decode=parallel_video_decode
        )
        video_config.update_model_arch(video_settings)
        audio_config = MiniMaxH3AudioVAEConfig()
        audio_config.update_model_arch(audio_settings)
        with torch.device("cpu"):
            self.video = MiniMaxH3VideoVAE(video_config)
            self.audio = MiniMaxH3AudioVAE(audio_config) if load_audio else None
        video_weights = load_file(
            str(root / "FL2VA/video_vae/source/model.safetensors")
        )
        video_weights.pop("latents_mean", None)
        video_weights.pop("latents_std", None)
        load_model_state_dict(self.video, video_weights, strict=True, assign=True)
        self.video.eval().requires_grad_(False).to(device)
        if self.audio is not None:
            audio_weights = load_file(
                str(root / "audio_vae/diffusion_pytorch_model.safetensors")
            )
            audio_weights.pop("latents_mean", None)
            audio_weights.pop("latents_std", None)
            audio_weights = {
                name.replace(".weight_g", ".parametrizations.weight.original0").replace(
                    ".weight_v", ".parametrizations.weight.original1"
                ): tensor
                for name, tensor in audio_weights.items()
            }
            load_model_state_dict(self.audio, audio_weights, strict=True, assign=True)
            self.audio.eval().requires_grad_(False).to(device)
        else:
            pass
        self.video_decode_backend = video_decode_backend
        self.video_use_graph = video_use_graph
        self.streaming_decoders: dict[str, TileStreamingDecoder] = {}
        self.idle_streaming_decoder: TileStreamingDecoder | None = None
        if video_attention_backend != AttentionBackendEnum.TORCH_SDPA:
            # Select for decoder compute precision, independently of FP32 VAE weights.
            with set_default_torch_dtype(torch.bfloat16):
                for module in self.video.modules():
                    if isinstance(module, VideoAttention):
                        module.attn = USPAttention(
                            num_heads=module.heads,
                            head_size=module.dim_head,
                            causal=False,
                            required_attention_backend=video_attention_backend,
                            skip_sequence_parallel=True,
                        )
                    else:
                        pass
        else:
            pass
        self.device = torch.device(device)
        self.audio_decode_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device) if parallel_audio_decode else None
        )
        self.audio_decode_executor = (
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="realization-audio-decode"
            )
            if parallel_audio_decode
            else None
        )
        if prepare_video_weights:
            self.video.prepare_decoder_autocast_weights(torch.float16)
        else:
            pass
        self.video_mean = torch.tensor(
            video_settings["latents_mean"], device=device
        ).reshape(1, 24, 1, 1, 1)
        self.video_std = torch.tensor(
            video_settings["latents_std"], device=device
        ).reshape(1, 24, 1, 1, 1)
        self.audio_mean = torch.tensor(
            audio_settings["latents_mean"], device=device
        ).reshape(1, 32, 1)
        self.audio_std = torch.tensor(
            audio_settings["latents_std"], device=device
        ).reshape(1, 32, 1)

    @torch.inference_mode()
    def encode_reference(
        self, pixels: torch.Tensor, *, seed: int, mode: DuoDecodeMode = "duo_1_4"
    ) -> torch.Tensor:
        if mode == "duo_5_17":
            with torch.autocast("cuda", enabled=False):
                image = pixels.to(self.device, dtype=torch.float32)
                image = image.permute(2, 0, 1)[None, :, None] / 255
                image = self.video.processor.transform_tensor(image, runtime_owned=True)
                moments = self.video._adaptive_encode(image)  # Upstream VAE API.
                mean = DiagonalGaussianDistribution(moments).mean.float()
            return (
                ((mean - self.video_mean) / self.video_std)[0].cpu().to(torch.bfloat16)
            )
        else:
            pass
        with (
            torch.random.fork_rng(devices=[self.device]),
            torch.autocast("cuda", dtype=torch.float16),
        ):
            torch.manual_seed(seed)
            latent = self.video.encode_images([pixels.cpu().numpy()])[0].float()
        return (
            ((latent[None] - self.video_mean) / self.video_std)[0]
            .cpu()
            .to(torch.bfloat16)
        )

    @torch.inference_mode()
    def decode_audio(self, audio_latents: torch.Tensor) -> torch.Tensor:
        assert self.audio is not None
        audio_stream = self.audio_decode_stream or torch.cuda.current_stream(
            self.device
        )
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(audio_stream),
            torch.jit.optimized_execution(False),
        ):
            return self.audio.decode(audio_latents).squeeze(1).float()

    def shutdown(self) -> None:
        if self.audio_decode_executor is not None:
            self.audio_decode_executor.shutdown(wait=True)
        else:
            pass

    @torch.inference_mode()
    def decode(
        self,
        reference: torch.Tensor,
        previous: list[torch.Tensor],
        current_video: torch.Tensor,
        current_audio: torch.Tensor,
        *,
        audio_samples: int,
        mode: DuoDecodeMode = "duo_1_4",
        session_id: str = "default",
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        audio_latents = (
            current_audio.to(self.device).float() * self.audio_std + self.audio_mean
        )
        caller_stream = torch.cuda.current_stream(self.device)
        audio_stream = self.audio_decode_stream
        if audio_stream is not None:
            audio_stream.wait_stream(caller_stream)
            audio_latents.record_stream(audio_stream)
        else:
            audio_stream = caller_stream
        if self.audio_decode_executor is None:
            wave = self.decode_audio(audio_latents)
            audio_future = None
        else:
            audio_future = self.audio_decode_executor.submit(
                self.decode_audio, audio_latents
            )
        try:
            if mode == "duo_1_4":
                frames = self.decode_1_4(reference, previous, current_video)
            elif self.video_decode_backend == "solarx":
                frames = self.decode_video_stream(session_id, current_video)
            else:
                frames = self.decode_5_17(previous, current_video)
        finally:
            if audio_future is not None:
                wave = audio_future.result()
                caller_stream.wait_stream(audio_stream)
            else:
                pass
        wave.record_stream(caller_stream)
        if mode == "duo_1_4" and (
            wave.shape != (2, 5600) or audio_samples not in (5333, 5334)
        ):
            raise ValueError("Invalid four-frame audio decode")
        else:
            pass
        if mode == "duo_5_17" and (wave.shape[0] != 2 or wave.shape[1] < audio_samples):
            raise ValueError("Invalid seventeen-frame audio decode")
        else:
            pass
        return frames, wave[:, :audio_samples].cpu().contiguous()

    def decode_video_window(
        self, latents: list[torch.Tensor], crop: tuple[int, int]
    ) -> torch.Tensor:
        video_latents = torch.cat(latents, dim=1)[None].to(self.device).float()
        video_latents = video_latents * self.video_std + self.video_mean
        with set_forward_context(0, None), torch.autocast("cuda", dtype=torch.bfloat16):
            pixels = self.video.decode_temporal(video_latents)
        if not torch.isfinite(pixels).all() or pixels.shape[2] < crop[1]:
            raise ValueError("Invalid temporal video decode")
        else:
            pass
        pixels = self.video.processor.revert_tensor(
            pixels[:, :, crop[0] : crop[1]].float()
        )[0]
        return (pixels.permute(1, 2, 3, 0) * 255).round().clamp(0, 255).byte().cpu()

    def decode_1_4(
        self,
        reference: torch.Tensor,
        previous: list[torch.Tensor],
        current_video: torch.Tensor,
    ) -> torch.Tensor:
        window = ([reference] + previous[-2:] + [current_video])[-3:]
        video_latents = torch.cat(window, dim=1)[None].to(self.device).float()
        video_latents = video_latents * self.video_std + self.video_mean
        with set_forward_context(0, None), torch.autocast("cuda", dtype=torch.float16):
            pixels = self.video._adaptive_decode(video_latents)  # Upstream VAE API.
        if pixels.shape[2] != 4 * len(window) or not torch.isfinite(pixels).all():
            raise ValueError("Invalid temporal video decode")
        else:
            pass
        pixels = self.video.processor.revert_tensor(pixels.float())[0, :, -4:]
        return (pixels.permute(1, 2, 3, 0) * 255).round().clamp(0, 255).byte().cpu()

    def decode_5_17(
        self, previous: list[torch.Tensor], current_video: torch.Tensor
    ) -> torch.Tensor | None:
        if not previous:
            return None
        else:
            pass
        if len(previous) == 1:
            window = previous[-1:] + [current_video[:, :2]]
            return self.decode_video_window(window, (0, 17))
        else:
            pass
        window = previous[-2:] + [current_video[:, :2]]
        return self.decode_video_window(window, (17, 34))

    @torch.inference_mode()
    def finish_5_17(self, previous: list[torch.Tensor]) -> torch.Tensor | None:
        if not previous or self.video_decode_backend == "solarx":
            return None
        else:
            pass
        window = previous[-2:] + [previous[-1][:, -2:]]
        crop = (0, 17) if len(previous) == 1 else (17, 34)
        return self.decode_video_window(window, crop)

    @property
    def video_lookahead_blocks(self) -> int:
        return int(self.video_decode_backend == "native")

    def close_stream(self, session_id: str) -> None:
        decoder = self.streaming_decoders.pop(session_id, None)
        if decoder is not None:
            decoder.reset()
            self.idle_streaming_decoder = decoder
        else:
            pass

    @torch.inference_mode()
    def decode_video_stream(
        self, session_id: str, latents: torch.Tensor
    ) -> torch.Tensor:
        with torch.cuda.device(self.device):
            if session_id not in self.streaming_decoders:
                if self.idle_streaming_decoder is not None:
                    decoder = self.idle_streaming_decoder
                    self.idle_streaming_decoder = None
                else:
                    self.video.decoder.to(dtype=torch.bfloat16)
                    self.video.post_quant_conv.to(dtype=torch.bfloat16)
                    _, tile_ranks = get_tile_parallel_state()
                    group = (
                        get_decode_parallel_group_coordinator()
                        if tile_ranks > 1
                        else None
                    )
                    decoder = TileStreamingDecoder(
                        self.video, group=group, use_graph=self.video_use_graph
                    )
                self.streaming_decoders[session_id] = decoder
            else:
                decoder = self.streaming_decoders[session_id]
            normalized = (
                latents[None].to(self.device).float() * self.video_std + self.video_mean
            )
            with torch.autocast("cuda", enabled=False):
                pixels = torch.cat(
                    [
                        decoder.push(normalized[:, :, index : index + 1])
                        for index in range(latents.shape[1])
                    ],
                    dim=2,
                )
                pixels = self.video.processor.revert_tensor(pixels.float())[0]
                frames = (pixels.permute(1, 2, 3, 0).clamp(0, 1) * 255).round().byte()
            return frames.cpu().contiguous()
