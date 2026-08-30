# Negative results

Every approach below was tried on the machine and measured. They are recorded
at the same level of detail as the ones that shipped, because on this
hardware most of the plausible ideas lose — and knowing *why* they lose is
what made the winning configuration findable.

## 1. llama.cpp is faster on paper than in a chat

`llama-bench` reports **54.6 tok/s** for `tg64` at depth 0 with the official
MXFP4 GGUF — better than the SGLang production stack. Through
`llama-server`'s chat endpoint with the same benchmark protocol used
everywhere else, it delivers **50.4 tok/s**, i.e. below production. The
synthetic generation benchmark does not include the serving path.

Adding llama.cpp's Eagle3 draft GGUF (`--spec-draft-n-max 3`) changed nothing
(50.3 tok/s); the log showed `eagle3 requires ctx_other to be set` during
memory fitting and speculation never engaged in the chat endpoint.

## 2. SGLang + Eagle3 is currently impossible on GB10

A genuine dependency deadlock, verified across three images:

- `lmsysorg/sglang:spark` (0.5.4) has the only `triton_kernels` build that
  still runs MXFP4 on `sm_121`, but its gpt-oss Eagle3 wiring feeds the draft
  head 2 concatenated hidden states where NVIDIA's head expects 3 →
  `mat1 and mat2 shapes cannot be multiplied (6x5760 and 8640x2880)` on the
  first request.
- `0.5.17` and the `dev-cu13` nightly fix the Eagle3 side but their newer
  `triton_kernels` refuses `sm_121` outright:
  `Must use persistent kernel and be TMA-compliant for native MXFP`.
  `--disable-prefill-cuda-graph` only moves the crash from capture to decode.
- Their `flashinfer_mxfp4` fallback runs but delivers **34.0 tok/s** — a
  regression, since those kernels target datacenter Blackwell (`sm_100`).

## 3. Newer is not faster: SGLang nightly

`dev-cu13` (0.5.17+) on the production weights: **34.0 tok/s** versus 52.6
for the year-old `:spark` image. Upstream moved away from this chip.

## 4. Every stock vLLM MoE backend

| backend | result |
| --- | --- |
| `marlin` | 34.7 tok/s |
| `humming` | 35.1 tok/s stock; **33.2 tok/s even with our dense quantization** |
| `b12x` | rejected: *kernel does not support expert biases* (gpt-oss needs them) |
| `triton` / `triton_unfused` | rejected: *does not support current device* |
| `flashinfer_trtllm` | no `sm_121` cubins exist in `flashinfer_cubin` |
| `flashinfer_cutlass` | rejected on the default activation key until [patches/01](../patches/01-moe-backend-selection.patch) |

`humming` deserved the closest look — it JIT-compiles to `sm_121a`, hardcodes
GB10's bandwidth, and avoids the 128-alignment padding entirely. It is still
half the speed of the CUTLASS path on these shapes.

## 5. Speculation depth K≥2

Counter-intuitive and worth internalising: on a MoE model, verifying N tokens
costs nearly N× the *expert* traffic, because different tokens route to
different experts. Verification is not "almost free" the way it is on a dense
model.

| K | tokens/pass | pass | throughput |
| ---: | ---: | ---: | ---: |
| 1 | 1.61 | 24.7 ms | **65.2 tok/s** |
| 2 | 1.96 | 32.3 ms | 60.7 tok/s |

A second draft step costs 7.6 ms and returns ~0.35 accepted tokens. It also
drops GPU utilisation from ~100 % to 89.5 %, i.e. it leaves the CUDA-graph
path — a real upstream bug for anyone who needs K≥2, but irrelevant once you
accept K=1.

## 6. Retraining the Eagle3 draft head

Two full rounds, both instrumented end to end (capture hook → harmony-exact
token IDs → training → serving eval).

- **Round 1** on Wikipedia/FineWeb text: offline agreement 32.3 % → 44.3 %,
  serving throughput and acceptance **unchanged**. Distribution mismatch: the
  serving acceptance is measured on model-generated chat, not on encyclopedic
  prose.
- **Round 2** on 339k tokens the champion generated itself, rendered through
  `openai_harmony` into exact token IDs: offline agreement 61.0 % → **67.7 %**,
  serving still **neutral** (48.6 % vs 52.7 % acceptance, within run-to-run
  noise of ±2–3 points).

The measured reason: position-1 acceptance (the *recursive* second draft step)
got slightly worse. Single-step training on target hidden states cannot
improve the drafter's own recursion; that needs multi-step training. The
pipeline is kept because it is validated and reusable, not because it paid off.

## 7. The "12 ms host overhead" that never existed

Claimed three times on the strength of a `py-spy` profile showing 7.6 ms per
pass inside `torch.cuda.synchronize`, and refuted three times:

1. GPU utilisation sampled at 100 ms during speculative decoding: **89.5 %**.
2. `nsys` with `--cuda-graph-trace=node` over 998 passes: kernel time is
   **24.9 s of a 25 s window**.
3. The per-layer micro-benchmark accounts for the pass within ~5 %.

A Python profiler shows *which function the CPU sits in*, never whether it is
working or waiting on the device. Do not size an optimization from one.

## 8. MoE kernel configuration — and why the tile sweep proved nothing

Both knobs were swept at production shapes (128 experts, top-4, 2944 padded
dims), not at the synthetic shapes the upstream notes used:

- **Tiles**: default `64x128` → 190 GB/s. `128x128` → 156 GB/s. `64x64` fails
  to initialise (requests 102400 B of shared memory; GB10 allows 101376 B).
  `64x32` and smaller crash with `illegal instruction`.
- **Pipeline stages**: the CUTLASS auto-carveout picks 3. Forcing 2 makes the
  kernel **2.2× slower** (83 GB/s) — this kernel lives on prefetch depth, not
  on occupancy. 4+ does not fit in shared memory.

**The tile row above is not a tuning result.** `compute-sanitizer` on `64x64`
shows the host launcher instantiated with `TileShape = <128,128,128>` while the
device kernel it launches was compiled as `<64,64,128>`: the fork's
`-DLOGICAL_TILE_M/N` flags rewrite the device-side namespace, but the host
still picks its tile from a heuristic offering only four SM120 shapes. Grid and
tile counts therefore disagree for every tile except `64x128` — which survives
only because at M ≤ 64 it produces the same counts as `128x128`. Six "crashing
tiles" are one bug seen six times. Two further blockers were found and fixed
(an `EPI_TILE_N must divide CTA_N` static assert in the transposed path;
`StageCountAutoCarveout` overshooting the smem limit by exactly 1 KB for
`64x64`), so those two now compile and reach the same crash. A real tile study,
including the transposed decode path the fork advertises, has to make the host
dispatch agree with the compiled tile first.

## 8b. Cooperative instead of Pingpong: no effect

The launcher hardcodes `KernelPtrArrayTmaWarpSpecializedPingpong` and its own
comment flags `KernelScheduleAuto` as untested. Pingpong splits the M tile
across two math warpgroups, which looks wasteful when M is 1–2 tokens, so
Cooperative was built and measured. Within noise at every expert count from 4
to 8, never more than 1.5 % apart.

Worth recording *how* this was nearly a false positive: the first comparison,
using the stock harness at 30 iterations, showed Cooperative 5–6 % ahead. Two
things were wrong with it. The harness silently could not force a union of more
than `TOPK` experts (it sliced the pool back down to 4), and single 30-iteration
timings on this box scatter by ±5 % — the same size as the claimed effect.
`bench/moe_sched_bench.py` builds the routing explicitly and reports the min of
5×60 iterations; the effect disappeared.

## 9. CUTLASS's block-scaled GEMV cannot be used

`gemm/kernel/gemv_blockscaled.h` exists in the vendored CUTLASS and looks like
the ideal small-M path. Four independent blockers: `static_assert(kSFVecSize == 16)`
(MXFP4 needs 32), scale type `float_e4m3_t` (we need `ue8m0`),
`kDequantizeA && kDequantizeB` (our activations are FP8, not FP4), and a grid
that assumes `N == 1` with no expert grouping. Usable as a blueprint for a
hand-written kernel, not as a drop-in.

## 10. Assorted smaller ones

- **Marlin `lm_head` suspected, acquitted**: 1.33 ms at **93 %** of its
  bandwidth floor. The bf16 *draft MLP* next to it was the real cost (1.5 ms),
  found by per-layer micro-benchmark rather than by intuition.
- **`--async-scheduling`**: flat, in every configuration tested.
- **CUDA-graph mode**: `PIECEWISE` for the target is 31.4 tok/s versus 58.5
  for `FULL_AND_PIECEWISE` with `TRITON_ATTN`.
- **The "+13.8 % padding" claim** from a survey of the code does not apply to
  this path: `flashinfer_cutlass` rounds 2880 → 2944 (128-alignment, +4.5 %
  bytes), not → 3072. Only Marlin/TRT-LLM round to 256.
- **A near-miss**: the draft `lm_head` appeared unquantized because
  `logger.info_once` deduplicated the second, textually identical log line.
  Logging the module id showed two distinct modules, both quantized. A "fix"
  for this non-bug was one commit away.
