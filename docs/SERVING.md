# Serving

## 1. Two profiles, and how to choose

The image ships one engine and two configurations. They are not
interchangeable — the fastest single-user setup is the *wrong* choice for a
classroom, and vice versa.

| | `PROFILE=spec` | `PROFILE=plain` |
| --- | --- | --- |
| single stream | **65.2 tok/s** | 62.4 tok/s |
| concurrent slots | 2 | 32 |
| 30 users, aggregate | not applicable | **299.5 tok/s** |
| 30 users, TTFT median | — | 1.2 s |
| extra requirement | Eagle3 head mounted at `/eagle` | — |

`spec` runs Eagle3 speculative decoding at K=1. Speculation trades extra
compute for fewer sequential steps, which only pays when the machine is
otherwise idle — under concurrent load the same silicon is better spent on
other users' tokens. Anything above ~2 concurrent users should run `plain`.

```bash
# single user, maximum speed
docker run --gpus all --network host --ipc host --shm-size 32g \
  -v /srv/models/gpt-oss-120b:/model:ro \
  -v /srv/models/gpt-oss-120b-Eagle3-v3:/eagle:ro \
  -v /srv/tiktoken:/tiktoken:ro -v gptoss-jit:/root/.cache/flashinfer \
  -e PROFILE=spec ghcr.io/OWNER/gptoss-spark:VERSION

# many users
docker run --gpus all --network host --ipc host --shm-size 32g \
  -v /srv/models/gpt-oss-120b:/model:ro \
  -v /srv/tiktoken:/tiktoken:ro -v gptoss-jit:/root/.cache/flashinfer \
  -e PROFILE=plain ghcr.io/OWNER/gptoss-spark:VERSION
```

## 2. Knobs

| variable | default | meaning |
| --- | --- | --- |
| `PROFILE` | `plain` | `plain` or `spec` |
| `MODEL_DIR` | `/model` | gpt-oss-120b MXFP4 checkpoint |
| `EAGLE_DIR` | `/eagle` | Eagle3 head (`spec` only) |
| `PORT` | `8100` | OpenAI-compatible endpoint |
| `SERVED_NAME` | `gptoss` | model id in the API |
| `GPU_MEM_UTIL` | `0.70` | 0.70 is the sweet spot: weights 61.3 GiB, KV 273k tokens, ~83 GiB cap. 0.65 still works (KV 158k, −5 % speed); 0.62 fails to allocate KV blocks |
| `MAX_MODEL_LEN` | `32768` | context window |
| `MAX_NUM_SEQS` | 32 / 2 | concurrent slots (per profile) |
| `SPEC_K` | `1` | draft depth; 2 measured slower, see [SPECULATION.md](SPECULATION.md) |
| `VLLM_SPARK_DENSE` | `qkv,o,lm_head,mlp,fc` | which dense layers get MXFP4; empty disables the largest win |

Everything after the image name is passed through to `vllm serve`.

## 3. First start

The SM121 kernels are JIT-compiled on first use — expect ~10 minutes before
the endpoint answers, plus ~6 minutes of weight loading. Mount a volume at
`/root/.cache/flashinfer` and every later start is warm (~7 min, dominated by
weight loading and CUDA-graph capture).

Memory: the model needs ~83 GiB with the default `GPU_MEM_UTIL`. On a 121 GB
Spark that leaves room for one more small service, not for a second LLM.

## 4. Verifying a deployment

```bash
bench/bench.py --base-url http://127.0.0.1:8100/v1 --model gptoss \
  --prompts bench/prompts.jsonl --requests 4 --concurrency 1 --max-tokens 512
bench/loadtest.py http://127.0.0.1:8100/v1 gptoss 30      # aggregate + TTFT
```

Expect the numbers in [RESULTS.md](RESULTS.md) within a few percent. A large
miss usually means the MoE backend fell back — check the log for
`Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend` and
`SPARK: using flashinfer_spark cutlass_fused_moe`.

## 5. Open items before this replaces a production stack

Stated plainly, because they are not done:

- **Tool-call and reasoning-parser parity.** The endpoint speaks the
  OpenAI-compatible protocol and streams `reasoning` deltas, but a stack
  migrating off SGLang should verify its own tool-call paths against this one.
- **A quality gate on your own evaluation set.** The 20-prompt comparison in
  [../README.md §4](../README.md) shows answer-level parity; it is not a
  substitute for a domain evaluation.
- **Supply chain.** The kernels come from a personal fork pinned by commit.
  Vendor or mirror it before depending on it in production.
- **Patch drift.** The patches are diffs against one nightly. Pin the image
  by digest; refresh the patches deliberately, not implicitly.
