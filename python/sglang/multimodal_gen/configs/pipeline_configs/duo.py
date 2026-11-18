"""Realtime Realization block layouts shared by the adapter and native codecs."""

from dataclasses import dataclass
from typing import Literal

DuoDecodeMode = Literal["duo_1_4", "duo_5_17"]


@dataclass(frozen=True)
class DuoDecodeSpec:
    name: DuoDecodeMode
    input_frames: int
    output_frames: int
    video_latents: int
    audio_latents_per_channel: int
    input_audio_samples: int
    output_audio_samples: int
    duration_ms: float


DUO_DECODE_SPECS: dict[DuoDecodeMode, DuoDecodeSpec] = {
    "duo_1_4": DuoDecodeSpec(
        name="duo_1_4",
        input_frames=4,
        output_frames=4,
        video_latents=1,
        audio_latents_per_channel=7,
        input_audio_samples=round(16000 * 4 / 24),
        output_audio_samples=round(32000 * 4 / 24),
        duration_ms=1000 * 4 / 24,
    ),
    "duo_5_17": DuoDecodeSpec(
        name="duo_5_17",
        input_frames=17,
        output_frames=17,
        video_latents=5,
        audio_latents_per_channel=29,
        input_audio_samples=round(16000 * 17 / 24),
        output_audio_samples=round(32000 * 17 / 24),
        duration_ms=1000 * 17 / 24,
    ),
}


def resolve_duo_decode_mode(value: object) -> DuoDecodeMode:
    if value in (None, "old", "duo_1_4"):
        return "duo_1_4"
    elif value == "duo_5_17":
        return "duo_5_17"
    else:
        raise ValueError("sglang.realization.decode must be one of duo_1_4 or duo_5_17")
