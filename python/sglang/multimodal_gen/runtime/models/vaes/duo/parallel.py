"""Persistent workers for the native Realization VAE's overlapping spatial tiles."""

import atexit
import gc
import os
import threading
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import torch
import torch.multiprocessing as multiprocessing

from sglang.multimodal_gen.configs.pipeline_configs.duo import DuoDecodeMode
from sglang.multimodal_gen.runtime.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend,
)
from sglang.multimodal_gen.runtime.models.vaes.duo.codec import DuoCodecs
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.srt.utils import freeze_gc


@dataclass(kw_only=True)
class VideoDecodeRequest:
    latents: list[torch.Tensor]
    crop: tuple[int, int] = (0, 17)
    session_id: str = "default"


@dataclass(kw_only=True)
class VideoStreamClose:
    session_id: str


@dataclass(kw_only=True)
class VideoDecodeReply:
    frames: torch.Tensor | None = None
    error: str | None = None


def run_video_decode_rank(
    rank: int,
    devices: tuple[int, ...],
    model_path: str,
    rendezvous: str,
    connection: Connection,
    attention_backend: AttentionBackendEnum,
    decode_backend: Literal["native", "solarx"],
    use_graph: bool,
) -> None:
    device = torch.device("cuda", devices[rank])
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(devices[rank]),
        WORLD_SIZE=str(len(devices)),
    )
    torch.cuda.set_device(device)
    try:
        set_global_server_args(ServerArgs.from_dict({"model_path": model_path}))
        global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
        init_distributed_environment(
            world_size=len(devices),
            rank=rank,
            local_rank=devices[rank],
            device_id=device,
            distributed_init_method=rendezvous,
            backend="nccl",
        )
        initialize_model_parallel(
            sequence_parallel_degree=len(devices),
            ulysses_degree=len(devices),
            backend="nccl",
        )
        codecs = DuoCodecs(
            model_path,
            device=str(device),
            parallel_video_decode=True,
            video_decode_backend=decode_backend,
            video_use_graph=use_graph,
            load_audio=False,
            video_attention_backend=attention_backend,
        )
        codecs.video.prepare_decoder_autocast_weights(torch.bfloat16)
        gc.collect()
        freeze_gc(f"VAE tile rank {rank}")
        connection.send(VideoDecodeReply())
        with torch.inference_mode():
            while True:
                command: VideoDecodeRequest | VideoStreamClose | None = (
                    connection.recv()
                )
                if command is None:
                    break
                elif isinstance(command, VideoStreamClose):
                    codecs.close_stream(command.session_id)
                    connection.send(VideoDecodeReply())
                    continue
                else:
                    pass
                if decode_backend == "solarx":
                    frames = codecs.decode_video_stream(
                        command.session_id, command.latents[0]
                    )
                else:
                    frames = codecs.decode_video_window(command.latents, command.crop)
                connection.send(VideoDecodeReply(frames=frames if rank == 0 else None))
    except BaseException:
        connection.send(VideoDecodeReply(error=traceback.format_exc()))
        raise
    finally:
        connection.close()
        if torch.distributed.is_initialized():
            cleanup_dist_env_and_memory()
        else:
            pass


class ParallelDuoCodecs(DuoCodecs):
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        devices: tuple[int, ...],
        video_attention_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
        video_decode_backend: Literal["native", "solarx"] = "native",
        video_use_graph: bool = True,
    ) -> None:
        if len(devices) not in (2, 4) or len(set(devices)) != len(devices):
            raise ValueError(
                "Realization VAE workers require two or four distinct GPU indices"
            )
        else:
            pass
        super().__init__(
            model_path,
            device=device,
            parallel_audio_decode=True,
            video_decode_backend=video_decode_backend,
        )
        self.video.to("cpu")
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        self.connections: list[Connection] = []
        self.workers: list[multiprocessing.Process] = []
        self.decode_lock = threading.Lock()
        self.rendezvous = TemporaryDirectory(prefix="realization-vae-ranks-")
        context = multiprocessing.get_context("spawn")
        atexit.register(self.shutdown)
        try:
            for rank in range(len(devices)):
                parent, child = context.Pipe()
                worker = context.Process(
                    target=run_video_decode_rank,
                    args=(
                        rank,
                        devices,
                        model_path,
                        (Path(self.rendezvous.name) / "group").as_uri(),
                        child,
                        video_attention_backend,
                        video_decode_backend,
                        video_use_graph,
                    ),
                    daemon=True,
                )
                worker.start()
                child.close()
                self.connections.append(parent)
                self.workers.append(worker)
            self.receive()
        except BaseException:
            self.shutdown(abort=True)
            raise

    @torch.inference_mode()
    def encode_reference(
        self, pixels: torch.Tensor, *, seed: int, mode: DuoDecodeMode = "duo_1_4"
    ) -> torch.Tensor:
        # Decode workers own GPU video weights; the parent only encodes at open.
        with self.decode_lock, torch.cuda.device(self.device):
            self.video.to(self.device)
            try:
                return super().encode_reference(pixels, seed=seed, mode=mode)
            finally:
                self.video.to("cpu")
                torch.cuda.empty_cache()

    def receive(self) -> list[VideoDecodeReply]:
        replies = []
        for connection in self.connections:
            ready = wait(
                [connection, *(worker.sentinel for worker in self.workers)], timeout=600
            )
            if connection not in ready:
                raise RuntimeError("Realization VAE rank exited or timed out")
            else:
                pass
            reply: VideoDecodeReply = connection.recv()
            if reply.error is not None:
                raise RuntimeError(reply.error)
            else:
                pass
            replies.append(reply)
        return replies

    def decode_video_window(
        self, latents: list[torch.Tensor], crop: tuple[int, int]
    ) -> torch.Tensor:
        with self.decode_lock:
            for connection in self.connections:
                connection.send(VideoDecodeRequest(latents=latents, crop=crop))
            frames = self.receive()[0].frames
        assert frames is not None
        return frames

    def decode_video_stream(
        self, session_id: str, latents: torch.Tensor
    ) -> torch.Tensor:
        with self.decode_lock:
            for connection in self.connections:
                connection.send(
                    VideoDecodeRequest(latents=[latents], session_id=session_id)
                )
            frames = self.receive()[0].frames
        assert frames is not None
        return frames

    def close_stream(self, session_id: str) -> None:
        if self.video_decode_backend == "solarx":
            with self.decode_lock:
                for connection in self.connections:
                    connection.send(VideoStreamClose(session_id=session_id))
                self.receive()
        else:
            pass

    def shutdown(self, *, abort: bool = False) -> None:
        if not abort:
            for connection in self.connections:
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError):
                    pass
            for worker in self.workers:
                worker.join(timeout=5)
        else:
            pass
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
            else:
                pass
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.kill()
                worker.join()
            else:
                pass
        self.workers.clear()
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.rendezvous.cleanup()
        super().shutdown()
