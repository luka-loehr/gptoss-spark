![gptoss-spark banner](docs/assets/banner.svg)

# gptoss-spark – gpt-oss-120b at 68.9 tok/s on one DGX Spark

[![Model](https://img.shields.io/badge/model-gpt--oss--120b%20MXFP4-76B900?style=flat)](https://huggingface.co/openai/gpt-oss-120b) [![Engine](https://img.shields.io/badge/engine-vLLM%20%2B%206%20patches-1f6feb?style=flat)](docs/PATCHES.md) [![Platform](https://img.shields.io/badge/platform-DGX%20Spark%20GB10%20(sm__121)-555555?style=flat)](docs/SERVING.md) [![License](https://img.shields.io/badge/License-Apache--2.0-orange?style=flat)](LICENSE)

**gptoss-spark** serves `gpt-oss-120b` on a single NVIDIA DGX Spark at 68.9 tokens/s single-stream and 300 tokens/s aggregate at 30 concurrent users, up from 52.6 and 244 on the previous SGLang stack. It is a Docker image built from six small patches on upstream vLLM and SM121-tuned CUTLASS MXFP4 kernels.

---

## Features

- **68.9 tok/s single-stream** with Eagle3 speculative decoding at K=1 (`PROFILE=spec`)
- **300 tok/s aggregate at 30 users**, 1.2 s median TTFT, 32 slots (`PROFILE=plain`, 64.3 tok/s single-stream)
- **MXFP4 dense layers** quantizes attention projections and `lm_head`, the largest single gain
- **SM121 CUTLASS MoE kernels** vendored alongside the engine's own FlashInfer
- **Answer quality checked** against the previous stack on 20 prompts, not assumed
- **Reproducible numbers** from the benchmark harness in [`bench/`](bench/) and raw results in [`results/`](results/)
- **Negative results documented** for every approach that was measured and did not ship

---

## Quick start

Requires a DGX Spark (GB10, `sm_121`), Docker with the NVIDIA runtime, the gpt-oss-120b MXFP4 checkpoint (not in the image) and ~90 GB of free memory.

```bash
docker run --gpus all --network host --ipc host --shm-size 32g \
  -v /srv/models/gpt-oss-120b:/model:ro \
  -v /srv/tiktoken:/tiktoken:ro \
  -v gptoss-jit:/root/.cache/flashinfer \
  -e PROFILE=plain ghcr.io/luka-loehr/gptoss-spark:0.2.0
```

Use `PROFILE=spec` with an Eagle3 head mounted at `/eagle` for single-user speed. The first start JIT-compiles the kernels (~10 min); the cache volume keeps later starts warm.

---

## Documentation

- [Results](docs/RESULTS.md) – what produced the speedup, every configuration measured, concurrency, memory, quality
- [Serving](docs/SERVING.md) – profiles, environment variables, verifying a deployment, building from source
- [Patches](docs/PATCHES.md) – what each patch changes and which upstream bug it works around
- [Kernels](docs/KERNELS.md) – nsys profile of a decode pass and what is left to optimize
- [Speculation](docs/SPECULATION.md) – Eagle3 on a MoE model, the K sweep, draft head retraining
- [Negative results](docs/NEGATIVE-RESULTS.md) – approaches that did not work, and why
- [Changelog](CHANGELOG.md)

---

## License

Apache-2.0 - [View License](LICENSE)  
The files in `patches/` are modified from [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0); the kernels are built from a fork of [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (Apache-2.0) and [CUTLASS](https://github.com/NVIDIA/cutlass) (BSD-3-Clause), see [NOTICE](NOTICE). Model weights are not included and keep their own licenses.

---

## Support

- [Report bugs](https://github.com/luka-loehr/gptoss-spark/issues)  
- [luka@lukaloehr.com](mailto:luka@lukaloehr.com)  

---

Developed by [Luka Löhr](https://github.com/luka-loehr)
