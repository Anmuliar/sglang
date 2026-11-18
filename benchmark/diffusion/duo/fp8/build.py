# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build the scoped H3 async FA3 extension from pinned upstream sources."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flash-root",
        type=Path,
        required=True,
        help="vLLM FlashAttention Git checkout",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and patch sources without compiling",
    )
    parser.add_argument(
        "--cutlass-root",
        type=Path,
        required=True,
        help="CUTLASS Git checkout at pinned revision",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Build output directory"
    )
    parser.add_argument("--jobs", type=int, default=8)
    build_options = parser.parse_args()
    if build_options.jobs < 1:
        parser.error("--jobs must be positive")
    cutlass = build_options.cutlass_root.resolve()
    expected = "62750a2b75c802660e4894434dc55e839f322277"
    try:
        revision = subprocess.run(
            ["git", "-C", str(cutlass), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(cutlass),
                "status",
                "--porcelain",
                "--",
                "include",
                "tools/util/include",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"cannot identify CUTLASS Git checkout: {exc}")
    if revision != expected or status:
        parser.error(
            f"CUTLASS must be clean at {expected}; found {revision}, dirty={bool(status)}"
        )
    if (
        not (cutlass / "include").is_dir()
        or not (cutlass / "tools/util/include").is_dir()
    ):
        parser.error("CUTLASS include paths are missing")
    flash = build_options.flash_root.resolve()
    revision = subprocess.run(
        ["git", "-C", str(flash), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(flash),
            "status",
            "--porcelain",
            "--",
            "hopper",
            "csrc/common",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if revision != "9cd61de38763d712bb6ce56e2a02cc2bf718c89f" or status:
        parser.error(
            f"FlashAttention must be clean at the pinned revision; found {revision}, dirty={bool(status)}"
        )
    output = build_options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir()
    hopper = source / "hopper"
    common = source / "common"
    shutil.copytree(flash / "hopper", hopper)
    shutil.copytree(flash / "csrc/common", common)
    patch = Path(__file__).resolve().parent / "csrc/fullapi_async.patch"
    subprocess.run(["patch", "-p1", "-i", str(patch)], cwd=hopper, check=True)
    if build_options.prepare_only:
        print(f"Prepared sources: {source}", flush=True)
        return
    output = output / "build"
    output.mkdir()
    os.environ["MAX_JOBS"] = str(build_options.jobs)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    if not CUDA_HOME:
        parser.error("CUDA_HOME is not configured")
    macros = [
        "NDEBUG",
        "FLASHATTENTION_DISABLE_BACKWARD",
        "FLASHATTENTION_DISABLE_DROPOUT",
        "FLASHATTENTION_DISABLE_UNEVEN_K",
        "FLASHATTENTION_DISABLE_PYBIND",
        "FLASHATTENTION_VARLEN_ONLY",
        "FLASHATTENTION_PACKGQA_ONLY",
        "FLASHATTENTION_DISABLE_CLUSTER",
        "FLASHATTENTION_DISABLE_SM8x",
        "FLASHATTENTION_DISABLE_APPENDKV",
        "FLASHATTENTION_DISABLE_SPLIT",
        "FLASHATTENTION_DISABLE_SOFTCAP",
        "FLASHATTENTION_DISABLE_LOCAL",
        "FLASHATTENTION_DISABLE_PAGEDKV",
        "FLASHATTENTION_DISABLE_FP16",
        "FLASHATTENTION_DISABLE_HDIM64",
        "FLASHATTENTION_DISABLE_HDIM96",
        "FLASHATTENTION_DISABLE_HDIM192",
        "FLASHATTENTION_DISABLE_HDIM256",
        "FLASHATTENTION_DISABLE_HDIMDIFF64",
        "FLASHATTENTION_DISABLE_HDIMDIFF192",
        "CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "CUTLASS_ENABLE_GDC_FOR_SM90",
        "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
        "TORCH_TARGET_VERSION=0x020A000000000000ULL",
        "USE_CUDA",
        "H3_TILE_N=128",
    ]
    flags = ["-D" + x for x in macros]
    sources = [
        hopper / "flash_api.cpp",
        hopper / "flash_api_torch_lib.cpp",
        hopper / "flash_prepare_scheduler.cu",
        hopper / "instantiations/flash_fwd_hdim128_e4m3_packgqa_sm90.cu",
    ]
    load(
        name="h3_stock_fa3_async_n128",
        sources=[str(x) for x in sources],
        extra_include_paths=[
            str(hopper),
            str(common),
            str(cutlass / "include"),
            str(cutlass / "tools/util/include"),
        ],
        extra_cflags=["-O3", "-std=c++20", *flags],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-Xptxas=-v",
            *flags,
        ],
        extra_ldflags=[
            "-Wl,-Bsymbolic",
            "-L" + str(Path(CUDA_HOME) / "compat"),
            "-lcuda",
        ],
        build_directory=str(output),
        verbose=True,
    )
    ns = torch.ops.h3_stock_fa3_async_n128
    print("forward:", ns.fwd.default._schema, flush=True)
    print("metadata:", ns.get_scheduler_metadata.default._schema, flush=True)
    print("module:", output / "h3_stock_fa3_async_n128.so", flush=True)


if __name__ == "__main__":
    main()
