"""Keep Realization gate ordering and Intent's BF16 rounding when switching kernels."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.duo.codec import DuoCodecs
from sglang.multimodal_gen.runtime.server_args import (
    server_args as diffusion_server_args,
)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_audio_decode_runs_while_video_decode_is_in_progress() -> None:
    """A separate stream alone serialized audio submission after video's CPU copy."""
    audio_started = threading.Event()
    video_started = threading.Event()
    codecs = DuoCodecs.__new__(DuoCodecs)
    codecs.device = torch.device("cuda:0")
    codecs.video_decode_backend = "native"
    codecs.audio_std = torch.ones(1, 32, 1, device="cuda")
    codecs.audio_mean = torch.zeros(1, 32, 1, device="cuda")
    codecs.audio_decode_stream = torch.cuda.Stream()
    codecs.audio_decode_executor = ThreadPoolExecutor(max_workers=1)
    expected_frames = torch.zeros(17, 2, 2, 3, dtype=torch.uint8)

    def audio_decode(audio_latents: torch.Tensor) -> torch.Tensor:
        assert torch.cuda.current_stream() == codecs.audio_decode_stream
        audio_started.set()
        assert video_started.wait(5), "video was not submitted concurrently"
        return audio_latents.mean(dim=1, keepdim=True)

    def video_decode(
        previous: list[torch.Tensor], current_video: torch.Tensor
    ) -> torch.Tensor:
        assert audio_started.wait(5), "audio was not submitted concurrently"
        video_started.set()
        return expected_frames

    codecs.audio = Mock()
    codecs.audio.decode.side_effect = audio_decode
    codecs.decode_5_17 = video_decode
    audio_latents = torch.ones(2, 32, 29)
    video_latents = torch.zeros(24, 5, 2, 2)
    try:
        frames, wave = codecs.decode(
            video_latents[:, :1],
            [],
            video_latents,
            audio_latents,
            audio_samples=8,
            mode="duo_5_17",
        )
        torch.testing.assert_close(frames, expected_frames)
        torch.testing.assert_close(wave, torch.ones(2, 8))
        assert wave.device.type == "cpu"
    finally:
        video_started.set()
        codecs.shutdown()


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or "DUO_REALIZATION_MODEL_PATH" not in os.environ,
    reason="requires CUDA and DUO_REALIZATION_MODEL_PATH",
)
def test_solarx_session_reset_reproduces_first_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = os.environ["DUO_REALIZATION_MODEL_PATH"]
    monkeypatch.setattr(
        diffusion_server_args,
        "_global_server_args",
        diffusion_server_args.ServerArgs.from_dict({"model_path": model_path}),
    )
    codecs = DuoCodecs(
        model_path, device="cuda:0", video_decode_backend="solarx", load_audio=False
    )
    generator = torch.Generator().manual_seed(42)
    latents = torch.randn(24, 5, 16, 16, generator=generator).bfloat16()
    try:
        first = codecs.decode_video_stream("first-call", latents)
        for _ in range(2):
            codecs.decode_video_stream("first-call", latents)
        codecs.close_stream("first-call")
        restarted = codecs.decode_video_stream("second-call", latents)
        assert first.shape == (17, 256, 256, 3)
        torch.testing.assert_close(restarted, first, rtol=0, atol=0)
        assert codecs.finish_5_17([latents]) is None
        codecs.close_stream("second-call")
    finally:
        codecs.shutdown()
