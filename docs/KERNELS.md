# Kernel analysis

Everything here was measured on the machine: `nsys` for the timeline,
`bench/moe_kernel_bench.py` for the MoE kernel in isolation at production
shapes, `bench/draft_microbench.py` for the per-layer decomposition.

## 1. Where a decode pass goes

`nsys profile -t cuda,nvtx --cuda-graph-trace=node`, 25 s window during
steady-state speculative decoding (K=1), 998 passes → **25.1 ms/pass**,
~687 kernels/pass:

| group | ms/pass | share | kernels |
| --- | ---: | ---: | ---: |
| MoE expert GEMM (CUTLASS grouped, MXFP4×MXFP8) | 15.57 | 62.4 % | 72 |
| Marlin dense projections + both lm_heads | 5.62 | 22.5 % | 79 |
| MoE support kernels | 2.17 | 8.7 % | 252 |
| Router GEMM (bf16 wmma) | 0.75 | 3.0 % | 36 |
| Attention + KV write | 0.52 | 2.1 % | 74 |
| Norms, elementwise | 0.29 | 1.1 % | 174 |

Sum of kernel time: **24.9 s of the 25 s window.** The device is never idle
waiting for the host. Any further gain has to come out of the kernels
themselves.

## 2. The MoE kernel in isolation

`bench/moe_kernel_bench.py` reproduces the serving shapes exactly (128
experts, top-4, hidden and intermediate padded to 2944, MXFP4 weights, MXFP8
activations, expert biases, SwiGLU α/β/limit) and reports achieved bandwidth
against the *actual number of distinct experts touched*:

| tokens | distinct experts | time | traffic | achieved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 328 µs | 55 MB | 168 GB/s |
| 2 | 8 | 590 µs | 110 MB | 187 GB/s |
| 4 | 16 | 1137 µs | 221 MB | 194 GB/s |
| 8 | 24 | 1663 µs | 332 MB | 199 GB/s |
| 1–8 | **4 (forced)** | **~325 µs** | 55 MB | ~170 GB/s |

The last row is the important one: with the token count varying 1→8 but the
expert set held fixed, the time does not move. **This kernel's cost is set by
how many distinct experts are touched, not by how many tokens are processed.**

That single fact explains the speculation results in
[SPECULATION.md](SPECULATION.md): verifying two tokens reads roughly twice
the expert weights, because two tokens route to two different top-4 sets.

## 3. Roofline — and a correction

The table above averages a fixed per-call cost into a per-expert rate, which
is what produced the "~76 % of achievable" figure this document used to carry.
Sweeping the *distinct expert count* one at a time separates the two terms
(`bench/moe_sched_bench.py`, min of 5×60 iterations — single 30-iteration
timings on this box scatter by ±5 %, the same size as the effects involved):

| distinct experts | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time | 307.6 µs | 368.5 | 433.9 | 494.9 | 556.3 |

That is a straight line: **≈59 µs fixed + ≈62 µs per expert.** One expert is
13.8 MB, so the marginal rate is 223 GB/s — **89 % of the 250 GB/s this part
achieves.** The expert GEMM is very nearly at the roofline and there is no
+10 tok/s hiding in it.

The fixed 59 µs is the real target: across 36 layers it is **2.1 ms of a
24.6 ms decode pass**, spent on six support kernels that move almost no data.
See §5.

Achievable bandwidth on this part is ~250 GB/s (measured: a dense bf16 GEMV
reaches 250.9 GB/s, the Marlin MXFP4 `lm_head` 233 GB/s).

## 4. Which knobs are exhausted

- **Tile shape**: swept at production shapes. Default `64x128` wins (190
  GB/s); `128x128` is worse (156). Everything else fails, and §6 explains why
  it is not a tuning result.
- **Pipeline stages**: CUTLASS's `StageCountAutoCarveout` picks 3 and that is
  optimal. Forcing 2 costs 2.2× (83 GB/s) — the kernel is prefetch-bound, not
  occupancy-bound, which also rules out the "get to 2 CTAs/SM" plan: fewer
  stages is exactly what would buy that occupancy, and it is much slower.
- **Mainloop schedule**: the fork hardcodes
  `KernelPtrArrayTmaWarpSpecializedPingpong` with a comment flagging
  `KernelScheduleAuto` as untested. Cooperative was built and measured: within
  noise at every expert count (4→8), never more than 1.5 % apart. No effect.
- **Alternative backends**: all measured, all worse — see
  [NEGATIVE-RESULTS.md §4](NEGATIVE-RESULTS.md).

## 5. Where the fixed 59 µs goes, and what was taken out of it

Per-layer device times from the same nsys capture, with launch geometry:

| kernel | grid × block | µs/layer |
| --- | --- | ---: |
| `doActivationKernel` | 384 × 256 | 20.06 |
| `expandInputRowsKernel` | 384 × 256 | 18.66 |
| `computeStridesTmaWarpSpecialized` | **1 × 128** | 8.53 |
| `finalizeMoeRouting` | 2 × 256 | 6.30 |
| `topkGating` | **1 × 32** | 2.78 |
| `fusedBuildExpertMapsSortFirstToken` | **1 × 32** | 2.30 |

Three of the six run in a single CTA on a 48-SM part. The two expensive ones
are not doing expensive work: both end with a loop over
`min_num_tokens_alignment * num_experts` padding slots — 128 × 128 = 16384
iterations — that reloads two `expert_first_token_offset` entries from global
memory on every pass. At decode batch sizes only a handful of experts are
routed to, so over 99 % of those iterations write nothing and exist purely to
discover that. That is the 38.7 µs.

[`patches/kernels/01-moe-sf-padding-loop.patch`](../patches/kernels/01-moe-sf-padding-loop.patch)
inverts the loop to run once per expert and flattens the (padding row, SF
element) space across the block. The set of writes is unchanged and the kernel
output is bit-identical to the stock build (verified by dumping both at fixed
inputs and comparing). Measured: **12–15 µs less per MoE call** across every
expert count, and end to end

| | before | after |
| --- | ---: | ---: |
| `PROFILE=plain`, single stream | 62.4 | **65.0** |
| `PROFILE=spec` K=1, single stream | 65.4 | **68.0–69.0** |
| 30 concurrent users, aggregate | 299.5 | 270–308 (unchanged, noisy) |

The win is confined to small batches, which is exactly what a *fixed*
per-layer cost predicts: at 30 users there is enough real work per layer that
39 µs of overhead disappears into it.

Still on the table, in order of size:

- The three single-CTA kernels above, ~13.6 µs/layer = 0.5 ms/pass.
- The router GEMM, §6 of [RESULTS.md](RESULTS.md): 0.74 ms/pass, and a
  40-line Triton replacement is 2× faster at M=2 and bit-identical. Needs
  custom-op plumbing to survive `torch.compile` and CUDA graphs.
- A hand-written weight-stationary MXFP4 GEMV. With the marginal rate now
  measured at 89 % of achievable, the upside is ~1.5 ms/pass, not the ~3 ms
  this document previously claimed — and it is still 1–3 weeks of risky work.
  Lower priority than it looked.

## 6. The tile sweep was measuring the wrong thing

`compute-sanitizer` on a non-default tile shows the host launcher instantiated
as

    sm120_mixed_input_moe_gemm_kernelLauncher<..., cute::tuple<C<128>, C<128>, C<128>>, ...>

while the device kernel it launches was compiled, from the JIT's
`-DLOGICAL_TILE_M/-DLOGICAL_TILE_N` flags, as `<C<64>, C<64>, C<128>>`. The
two sides of this fork's tile mechanism are decoupled: the `-D` flags rewrite
the device-side namespace, the host still picks its `CutlassTileConfigSM120`
from a heuristic whose SM120 switch offers only four shapes. `64x128` survives
only because at M ≤ 64 it produces the same tile counts as `128x128`.

So "everything smaller crashes with illegal instruction" was never a statement
about those tiles — it is one bug reproducing six times. Two further blockers
were found and fixed along the way (an `EPI_TILE_N must divide CTA_N` static
assert in the transposed path, and `StageCountAutoCarveout` overshooting GB10's
101376 B by exactly 1 KB for `64x64`), which is why those two now compile and
reach the same crash as the rest. Making the host dispatch agree with the
compiled tile is the prerequisite for any real tile study, including the
transposed decode path the fork advertises but never validated.
