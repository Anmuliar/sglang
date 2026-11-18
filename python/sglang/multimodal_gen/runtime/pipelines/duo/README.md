# Duo causal inference components

These components execute the Duo Realization transformer and Codec without depending on SGLang-Omni. A serving frontend owns Qwen/H50, ASR, input timing, session cancellation, and periodic history-reset policy. This package owns denoising, clean commits, KV storage and eviction, TP/SP execution, optional FP8 kernels, and native or SolarX video decoding.

## Installation and model dependencies

For the [Duo Omni serving branch](https://github.com/sgl-project/sglang-omni/tree/sglang-omni-duo-support), use the [duo-0.5.23 checkout](https://github.com/Anmuliar/sglang/tree/duo-0.5.23), based on the same SGLang release that Omni requires. Install this checkout with the diffusion extra. The checkpoint reader and exact Duo sequence/sampler contracts still require the matching model-provider source containing torchtitan and data; this component extraction does not replace that dependency. Set DUO_MODEL_SOURCE to that directory and add it to PYTHONPATH. No vLLM or SGLang-Omni import is required by these modules.

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.23+duo pip install -e 'python[diffusion]'
export DUO_MODEL_SOURCE=/path/to/Duo/third_party/duo
export PYTHONPATH="$PWD/python:$DUO_MODEL_SOURCE"
export TORCHTITAN_LIVE=1
```

Use matching Duo base, distill, adapter, and protocol assets. Preserve checkpoint RoPE, AdaLN, codec configuration, and weight dtype. FP8 Linear weights do not enable FP8 Attention; the latter requires a separately configured calibration and SM90 library. Optional library build sources are in benchmark/diffusion/duo/fp8.

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

Set DUO_REALIZATION_MODEL_PATH to a prepared Codec checkpoint to run the stream close/reopen test. Set DUO_REALIZATION_FP8_ATTENTION_LIBRARY to the matching SM90 shared library to run its optional test. Distributed tests require up to four CUDA devices. End-to-end reference comparison must hold weights, preprocessing, plan/H50, seed, sampling, runtime versions, and reset settings constant.
