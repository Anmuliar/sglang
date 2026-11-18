"""Duo block denoising, clean commits, and bounded causal history."""

import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch
from data.dataloader_realization import RealizationSequence
from data.dataloader_realization_sequence import block_seed
from data.inference_budget import budget_index, cost
from data.inference_options import ROLLING_MEDIA_CACHE_POLICY
from torchtitan.models.duo_realization.pipelines.inference import sample_sequence
from torchtitan.models.duo_realization.pipelines.streaming import (
    BudgetMediaSession,
    condition,
    shared_condition,
)
from torchtitan.models.duo_realization.schedulers.dmd import rollout_grid
from torchtitan.models.duo_realization.schedulers.flow import add_noise, to_x0

from sglang.multimodal_gen.runtime.models.dits.duo.native_dcp import (
    CachedTimestepModel,
    NativeDCPRealization,
)


class NativeBudgetMediaSession(BudgetMediaSession):
    def __init__(
        self,
        model: torch.nn.Module,
        features: dict[str, object],
        args: Namespace,
        device: torch.device,
        *,
        reference_window_chunks: int = 0,
    ) -> None:
        self.reference_window_chunks = reference_window_chunks
        self.defer_finite_checks = (
            isinstance(model, NativeDCPRealization)
            and model.options.defer_finite_checks
            and args.stage == "dmd"
        )
        if (
            isinstance(model, NativeDCPRealization)
            and model.options.cache_timestep_projections
            and args.stage == "dmd"
        ):
            schedule = rollout_grid(args.steps, shift=args.flow_shift, device=device)
            model = CachedTimestepModel(model, schedule)
        else:
            pass
        super().__init__(model, features, args, device)

        self.cuda_timings: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.pending_kv_reset = False

    def window_start(self, block: int) -> int:
        return max(
            self.start,
            super().window_start(block),
            (
                block + 1 - self.reference_window_chunks
                if self.reference_window_chunks
                else 0
            ),
        )

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            end.record(torch.cuda.current_stream(self.device))
            self.cuda_timings[name] = (start, end)

    def initialize(self, part: dict[str, object], metrics: dict[str, object]) -> None:
        if self.kv is not None:
            return
        else:
            pass
        full = self.sequence(part)
        prefix = full.select(lambda group: group["kind"] in ("reference", "shared"))
        prefix = replace(
            prefix,
            refiner_text=prefix.refiner_text[:, : len(self.shared["h50"])],
            refiner_groups=tuple(
                group for group in prefix.refiner_groups if group["kind"] == "shared"
            ),
        )
        with self.measure("cache_prefill_seconds"):
            _, self.kv = self.model(
                prefix,
                torch.full((2,), self.clean_time, device=self.device),
                commit=True,
            )

    def current(
        self, part: dict[str, object], metrics: dict[str, object]
    ) -> RealizationSequence:
        if self.failed or part["block"] != self.next_block:
            raise ValueError("Invalid/noncontiguous media cache session")
        else:
            pass
        block = part["block"]
        start = block if self.pending_kv_reset else self.window_start(block)
        self.initialize(part, metrics)
        previous = self.kv
        evicted = {
            group["block"]
            for group in previous.groups
            if group["kind"] == "clean" and group["block"] < start
        }
        started = time.monotonic()
        self.kv = previous.evict(start)
        self.start = start
        metrics.update(
            realization_kv_reset=self.pending_kv_reset,
            denoise_cache_rows=self.kv.length,
            cache_evict_seconds=time.monotonic() - started,
            cache_evicted_blocks=len(evicted),
            cache_evicted_block_ids=sorted(evicted),
            cache_evicted_rows=previous.length - self.kv.length,
        )
        self.pending_kv_reset = False
        return self.sequence(part).select(
            lambda group: group["kind"] in ("text", "noisy")
        )

    def commit(
        self,
        sequence: RealizationSequence,
        values: tuple[torch.Tensor, torch.Tensor],
        metrics: dict[str, object],
    ) -> None:
        # current() already selected and padded the complete block; no row selection is needed.
        clean = replace(
            sequence.with_media(*values),
            groups=tuple(
                dict(group, kind="clean") if group["kind"] == "noisy" else group
                for group in sequence.groups
            ),
            time_ids=torch.zeros_like(sequence.time_ids),
        )
        with self.measure("cache_commit_seconds"):
            _, self.kv = self.model(
                clean,
                torch.full((2,), self.clean_time, device=self.device),
                cache=self.kv,
                commit=True,
            )
        self.next_block += 1

    def sample_dmd(
        self, sequence: RealizationSequence, block: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, int | float | str | bool]]:
        current = sequence.select(
            lambda group: group["block"] == block and group["kind"] in ("text", "noisy")
        )
        times = torch.full(
            (len(sequence.seeds) + 1,), self.clean_time, device=sequence.device
        )
        schedule = rollout_grid(
            self.args.steps, shift=self.args.flow_shift, device=sequence.device
        )
        generator = torch.Generator(device=sequence.device).manual_seed(
            block_seed(self.args.seed, self.observations["sample_id"], block)
        )
        state = tuple(
            torch.randn(
                value.shape,
                device=value.device,
                dtype=torch.float32,
                generator=generator,
            )
            for value in (current.video, current.audio)
        )
        finite_predictions: list[torch.Tensor] = []
        for index in range(self.args.steps):
            times[1:] = schedule[index]
            prediction, _ = self.model(current.with_media(*state), times, cache=self.kv)
            finite_predictions.extend(
                torch.isfinite(value).all() for value in prediction
            )
            state = tuple(
                to_x0(value, velocity.float(), schedule[index])
                for value, velocity in zip(state, prediction)
            )
            if index + 1 < self.args.steps:
                state = tuple(
                    add_noise(
                        value,
                        torch.randn(
                            value.shape,
                            device=value.device,
                            dtype=value.dtype,
                            generator=generator,
                        ),
                        schedule[index + 1],
                    )
                    for value in state
                )
            else:
                pass
        finite = torch.stack(
            [*finite_predictions, *(torch.isfinite(value).all() for value in state)]
        )
        # note (kemiao): Reject invalid predictions before clean KV can be committed.
        if not bool(finite.all()):
            valid = finite.tolist()
            for index in range(self.args.steps):
                if not all(valid[2 * index : 2 * index + 2]):
                    raise FloatingPointError(
                        f"Nonfinite Realization prediction at sampling step {index}"
                    )
                else:
                    pass
            raise FloatingPointError("Nonfinite generated Realization latent")
        else:
            pass
        return state, dict(
            denoising_steps=self.args.steps,
            nfe=self.args.steps,
            flow_shift=self.args.flow_shift,
            solver="consistency_renoise",
            cache_mode="rebuild_per_block",
            sequence_rows=sequence.length,
            current_rows=current.length,
            h50_rows=sequence.refiner_text.shape[1],
            generated_history_blocks=block,
            cfg=False,
        )

    @torch.no_grad()
    def generate(
        self, features: dict[str, object], block: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, object], dict[str, object]]:
        if self.index is None:
            self.index = budget_index(
                features, self.args.height, self.args.width, self.geometry.name
            )
        else:
            pass
        if self.shared is None:
            self.shared = shared_condition(features)
        else:
            pass
        part = condition(features, block)
        part["position"] = self.position(block)
        metrics: dict[str, object] = {}
        self.cuda_timings.clear()
        try:
            current = self.current(part, metrics)
            with self.measure("denoise_seconds"):
                if self.sampler:
                    values, denoise = self.sampler(current, self.kv, block)
                elif self.defer_finite_checks:
                    values, denoise = self.sample_dmd(current, block)
                else:
                    values, denoise = sample_sequence(
                        self.model,
                        current,
                        stage=self.args.stage,
                        steps=self.args.steps,
                        shift=self.args.flow_shift,
                        seed=block_seed(
                            self.args.seed, self.observations["sample_id"], block
                        ),
                        history=self.kv,
                        clean_time=self.clean_time,
                        cm_sampling_mode=(
                            self.args.cm_sampling_mode
                            if self.args.stage == "cm"
                            else None
                        ),
                    )
            self.commit(current, values, metrics)
            height, width = self.observations["reference"].shape[-2:]
            video = (
                values[0]
                .reshape(self.geometry.latents, height // 2, width // 2, 24, 2, 2)
                .permute(3, 0, 1, 4, 2, 5)
                .reshape(24, self.geometry.latents, height, width)
            )
            audio = (
                values[1].reshape(2, self.geometry.audio_latents, 32).permute(0, 2, 1)
            )
            # The blocking output copies also complete preceding clean KV writes on this stream.
            latent = dict(
                video=video.cpu().contiguous(), audio=audio.cpu().contiguous()
            )
            metrics.update(
                {
                    name: start.elapsed_time(end) / 1000
                    for name, (start, end) in self.cuda_timings.items()
                }
            )
        except BaseException:
            self.failed = True
            self.kv = None
            raise
        metrics.update(
            denoise,
            cache_mode="block",
            media_cache_policy=ROLLING_MEDIA_CACHE_POLICY,
            cache_rows=self.kv.length,
            cache_bytes=self.kv.nbytes,
            current_rows=current.length,
            h50_rows=current.refiner_text.shape[1],
            cache_committed_block=block,
            generated_history_blocks=block,
            media_window=dict(
                start_block=self.start,
                stop_block=block + 1,
                frames=(block + 1 - self.start) * self.geometry.frames,
            ),
            training_equivalent_rows=cost(self.index, self.start, block + 1)[
                "padded_rows"
            ],
            timing_mode="cuda_event_rank_local",
        )
        self.states.append(dict(latent=latent, media_cache_condition=part))
        return latent, metrics, part


class TokenWindowMediaSession(NativeBudgetMediaSession):
    def __init__(
        self,
        model: torch.nn.Module,
        features: dict[str, object],
        args: Namespace,
        device: torch.device,
        window_tokens: int,
        *,
        reference_window_chunks: int = 0,
    ) -> None:
        super().__init__(
            model,
            features,
            args,
            device,
            reference_window_chunks=reference_window_chunks,
        )
        if window_tokens <= 0:
            raise ValueError("Realization KV window must be positive")
        else:
            self.window_tokens = window_tokens

    def window_start(self, block: int) -> int:
        current = self.index["blocks"][block]
        media_tokens = current["video_rows"] + current["stereo_audio_rows"]
        plan_tokens = current["packed_rows"] - 2 * media_tokens
        available_tokens = (
            self.window_tokens - self.index["shared_rows"] - plan_tokens - media_tokens
        )
        if available_tokens < 0:
            raise ValueError(
                "Realization window cannot fit reference, shared, plan and one media block"
            )
        else:
            history_blocks = available_tokens // media_tokens
            return max(
                self.start,
                block - history_blocks,
                (
                    block + 1 - self.reference_window_chunks
                    if self.reference_window_chunks
                    else 0
                ),
                0,
            )
