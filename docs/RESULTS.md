# Measurement campaign

Every number here was produced with `bench/bench.py` (4 requests × 512 tokens,
single stream, temperature 0, `reasoning_effort=low`, decode median) or
`bench/loadtest.py` (N concurrent streams, aggregate = tokens ÷ makespan) on
one DGX Spark, GB10, 121 GB, `sm_121`. Raw artifacts are in
[`../results/`](../results/).

## 1. Single stream, every engine tried

| # | configuration | decode | note |
| ---: | --- | ---: | --- |
| 1 | **+ MoE SF-padding patch + 32k draft vocab, K=1** | **68.8** | this repo, `PROFILE=spec` |
| 2 | **+ MoE SF-padding patch, plain** | **65.0** | this repo, `PROFILE=plain` |
| 3 | vLLM nightly + patches, Eagle3 K=1 | 65.2 | before the kernel patch |
| 4 | vLLM nightly + patches, plain | 62.4 | before the kernel patch |
| 5 | vLLM nightly + patches, Eagle3 K=2 | 60.7 | deeper speculation loses |
| 6 | fork image (`vllm-mxfp4-spark`, Jan-2026 vLLM) | 61.0 | where the kernels come from |
| 7 | SGLang `:spark` 0.5.4 | 52.6 | previous production |
| 8 | llama.cpp b6fdd0ac, MXFP4 GGUF | 50.4 | `llama-bench tg64` claims 54.6 |
| 9 | vLLM nightly, `flashinfer_cutlass`, no dense quant | 34.7 | patches 01+02 only |
| 10 | vLLM nightly, `marlin` | 34.7 | |
| 11 | vLLM nightly, `humming` | 35.1 | 33.2 even with dense quant |
| 12 | SGLang `dev-cu13` (0.5.17+) | 34.0 | newer, slower on this chip |
| 13 | NVIDIA vLLM 26.07 + Eagle3 D1 | 42.2 | earlier lab qualification |
| 14 | NVIDIA vLLM 26.07, no speculation | 37.3 | |
| 15 | TensorRT-LLM 1.3.0rc12 + Eagle | 20.2 | |

Rows 13–15 are from the qualification round that preceded this work; they used
the same benchmark and the same weights.

Row 1 is the median of three runs (69.0 / 68.1 / 68.8); rows 1 and 2 differ
from 3 and 4 only by
[`patches/kernels/01`](../patches/kernels/01-moe-sf-padding-loop.patch), which
is bit-identical in output, plus — for row 1 — a draft head whose lm_head was
cut to 32768 rows by [`ops/shrink-draft-vocab.py`](../ops/shrink-draft-vocab.py).
Neither can change what the model answers: the padding patch writes exactly the
same scale factors, and the shrunken draft vocabulary only narrows what the
drafter may *propose*, while the target still verifies over the full 201088.

## 2. Time to first token

| configuration | TTFT median |
| --- | ---: |
| this repo, plain | 0.21 s |
| this repo, spec K=1 | 0.24 s |
| SGLang production | 0.70 s |
| llama.cpp | 0.46 s |

## 3. Concurrency

`bench/loadtest.py`, N streaming requests fired simultaneously, 512 tokens
each. "Aggregate" is all completion tokens divided by the makespan of the
whole round — the honest measure of how much work the box does.

| users | | SGLang production | this repo (plain) |
| ---: | --- | ---: | ---: |
| 1 | aggregate | 52.3 tok/s | **59.4** |
| 1 | makespan | 9.8 s | **8.6 s** |
| 10 | aggregate | 143.5 tok/s | **173.0** |
| 10 | per user | 14.5 tok/s | **17.6** |
| 10 | TTFT median | 0.39 s | 0.56 s |
| 10 | makespan | 35.7 s | **29.6 s** |
| 30 | aggregate | 244.1 tok/s | **299.5** |
| 30 | per user | 16.6 tok/s | 10.2 tok/s |
| 30 | TTFT median | 16.4 s | **1.23 s** |
| 30 | makespan | 62.9 s | **51.3 s** |

The per-user column at 30 concurrent users favours the old stack only because
it ran `--max-running-requests 15`: fifteen users decoded quickly while the
other fifteen sat in a queue for a median 16.4 seconds. This stack admits all
thirty at once — everyone sees text after ~1.2 s — and still finishes the
round 11.6 s earlier.

## 4. Memory

| | weights | KV cache | cap |
| --- | ---: | ---: | ---: |
| SGLang production | — | — | 96.3 GiB |
| this repo (`GPU_MEM_UTIL=0.70`) | 61.3 GiB | 273k tokens (fp8) | 83 GiB |
| this repo (`0.65`) | 61.3 GiB | 158k tokens | 77 GiB |
| this repo (`0.62`) | — | — | fails: no KV blocks |

## 5. Quality

20 prompts (German/English, school subjects, code, arithmetic), greedy,
top-5 logprobs captured from both stacks
(`bench/quality_capture.py`, `bench/quality_compare.py`; artifacts
`results/quality-*.json`).

- Auto-graded items: **4/5 correct on both** stacks — including the same
  failure on one raw multiplication.
- Answer content: semantically equivalent throughout, differences are phrasing
  and formatting.
- Token-stream agreement: diverges within the first ~10 tokens, mean
  |Δlogprob| 0.062 on the agreed prefix. Expected: dense-layer quantization
  changes logits slightly, and gpt-oss's analysis channel amplifies an early
  near-tie into a different wording. Exact-token identity is not a meaningful
  gate for a quantization change; answer-level agreement is.

## 6. Kernel-level accounting

See [KERNELS.md](KERNELS.md). Summary: at the record configuration the GPU
executes kernels ~100 % of the wall time and 62.4 % of a pass is the MoE
expert GEMM — but that GEMM's *marginal* cost is 62 µs per expert touched,
i.e. **89 % of achievable bandwidth**, not the 76 % this document used to
claim. The 76 % averaged a fixed ~59 µs per-layer cost into the per-expert
rate. The fixed part is what remains worth attacking, and
[`patches/kernels/01`](../patches/kernels/01-moe-sf-padding-loop.patch)
already removes 12–15 µs of it.

What is left, sized from the same nsys capture:

| item | ms/pass | note |
| --- | ---: | --- |
| three single-CTA MoE support kernels | 0.49 | `computeStrides` (1×128 threads), `topkGating` and `fusedBuildExpertMaps` (1×32) on a 48-SM part |
| router GEMM | 0.74 | see below |
| remaining SF-padding cost | ~0.4 | the per-row loop, not the flat one |

The **router** (2880 → 128, bias) is `F.linear` on 737 KB of weights and gets
36 GB/s: cuBLAS picks an 8-CTA `cutlass_80_wmma` kernel that costs 5.3 µs at
M=1 but 14.6 µs at M=2 — and M=2 is exactly what K=1 speculation runs. A
40-line Triton kernel, one CTA per expert, is bit-identical to `F.linear` and
does not degrade from M=1 to M=2 (12.9 µs in both, versus 25.9 for `F.linear`
at M=2, eager timings on a contended box). Landing it needs custom-op plumbing
so it survives `torch.compile` and CUDA-graph capture.

## 7. Reproducing a row

```bash
PROFILE=spec docker run ... ghcr.io/luka-loehr/gptoss-spark:0.1.0      # row 1
PROFILE=plain docker run ... ghcr.io/luka-loehr/gptoss-spark:0.1.0     # row 2
SPEC_K=2 PROFILE=spec docker run ...                                # row 5
```

Rows 3 and 4 (the pre-patch numbers) need the 0.1.0 image, whose kernels are
unpatched.

Rows 5–13 need their own images; the exact tags and flags are listed in
[NEGATIVE-RESULTS.md](NEGATIVE-RESULTS.md).
