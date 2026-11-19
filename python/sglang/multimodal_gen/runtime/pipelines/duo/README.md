# Duo causal inference components

These components execute the Duo Realization transformer and Codec without depending on SGLang-Omni. A serving frontend owns Qwen/H50, ASR, input timing, session cancellation, and periodic history-reset policy. This package owns denoising, clean commits, KV storage and eviction, TP/SP execution, optional FP8 kernels, and native or SolarX video decoding.

For deployment with SGLang-Omni, use the [0.5.23 serving branch](https://github.com/Anmuliar/sglang/tree/duo-0.5.23), which carries these components on the SGLang release pinned by Omni. This development branch tracks the main-based implementation.

## Installation and model dependencies

Install this SGLang source checkout with the diffusion extra. The checkpoint reader and exact Duo sequence/sampler contracts still require the matching model-provider source containing torchtitan and data; this component extraction does not replace that dependency. Set DUO_MODEL_SOURCE to that directory and add it to PYTHONPATH. No vLLM or SGLang-Omni import is required by these modules.

```bash
pip install -e 'python[diffusion]'
export DUO_MODEL_SOURCE=/path/to/Duo/third_party/duo
export PYTHONPATH="$PWD/python:$DUO_MODEL_SOURCE"
export TORCHTITAN_LIVE=1
```

Use matching Duo base, distill, adapter, and protocol assets. Preserve checkpoint RoPE, AdaLN, codec configuration, and weight dtype. FP8 Linear weights do not enable FP8 Attention; calibrated FP8 Attention passes E4M3 Q/K/V and per-head descales to SGLang’s flash_attn_with_kvcache on SM90.

## H200 TP8 long-window FP8 attention

With attention calibration enabled, Duo can use the SM90a long-window build of
`sgl-fa3`. Install the tested wheel into the worker environment before starting
the service (CUDA 13.0, PyTorch 2.14.1, Linux x86-64):

```bash
python -m pip install "${WHEEL_DIR}/sgl_fa3-0.1.0+duotp8.n128-cp310-abi3-linux_x86_64.whl"
export SGLANG_USE_SGL_FA3_KERNEL=1
python -c 'from importlib.metadata import version; import sgl_fa3; print(version("sgl-fa3"), sgl_fa3.__file__)'
```

The `n128` build selects a 128×128 tile for dense, single-split FP8 media
attention at batch size 1, seven local heads of dimension 128, 2,048–2,304
query rows and 57,344–65,536 key rows. Other shapes retain the wheel's normal
dispatch. The published stock wheel also supports calibrated FP8 attention;
the long-window specialization requires the custom build.

## Ownership and execution

- runtime.models.dits.duo contains causal transformer adapters, mask/position layout, physical KV storage, attention workspaces, tensor/sequence parallel execution, and quantized kernels.
- runtime.pipelines.duo.loader loads and shards the Realization engine; session implements block sampling, clean commit, history eviction, and reset execution. The caller supplies the model-provider argument namespace and feature records.
- runtime.models.vaes.duo contains Codec loading, reference encoding, spatial tile workers, and the licensed SolarX temporal decoder with its compensation tables.
- configs.pipeline_configs.duo defines the 1-to-4 and 5-to-17 block geometry shared by model execution and frontends.

NativeBudgetMediaSession.generate consumes the next contiguous block's feature record and returns video/audio latents and timing/cache metrics. It performs the configured denoising evaluations followed by a clean commit. Historical clean KV is retained according to the configured window. Setting pending_kv_reset before generation evicts accumulated media history while preserving fixed reference/shared conditions; the caller decides when to set it. The generation index and positional timeline are not restarted.

A rank owns its physical model and KV shard. The frontend must issue operations to all participating ranks in the same order, drain or cancel active operations before releasing a session, and preserve current GPU overlap barriers. Extraction changes Python ownership, not the worker transport or number of CPU/GPU copies. These are component APIs for a persistent caller, not a registered standalone text-to-video generate pipeline or an HTTP endpoint.

## Verification

The tests preserve the original numerical tolerances for causal masking, TP/SP, clean KV, eviction, graph refresh, FP8 linear execution and Codec state. Run them without adding Omni to PYTHONPATH:

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/duo -q
```

Set DUO_REALIZATION_MODEL_PATH to a prepared Codec checkpoint to run the stream close/reopen test. Distributed tests require up to four CUDA devices. End-to-end reference comparison must hold weights, preprocessing, plan/H50, seed, sampling, runtime versions, and reset settings constant.
