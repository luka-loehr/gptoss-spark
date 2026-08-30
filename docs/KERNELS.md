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

## 3. Roofline

Achievable bandwidth on this part is ~250 GB/s (measured: a dense bf16 GEMV
reaches 250.9 GB/s, the Marlin MXFP4 `lm_head` 233 GB/s). The MoE kernel
streams at 187–199 GB/s, i.e. **~76 %**. Per-call fixed cost is ~50 µs, which
across 36 layers is ~1.8 ms/pass.

Closing the gap to ~95 % would save ≈3 ms/pass → ≈73 tok/s. Removing the 252
support kernels and the oversized router GEMM as well (2.9 ms) would approach
≈80 tok/s. Both require CUTLASS-level work.

## 4. Which knobs are exhausted

- **Tile shape**: swept at production shapes. Default `64x128` wins (190
  GB/s); `128x128` is worse (156); everything smaller either fails to
  initialise (shared-memory request 102400 B > 101376 B available) or crashes
  with `illegal instruction`.
- **Pipeline stages**: CUTLASS's `StageCountAutoCarveout` picks 3 and that is
  optimal. Forcing 2 costs 2.2× (83 GB/s) — the kernel is prefetch-bound, not
  occupancy-bound, which also rules out the "get to 2 CTAs/SM" plan: fewer
  stages is exactly what would buy that occupancy, and it is much slower.
- **Alternative backends**: all measured, all worse — see
  [NEGATIVE-RESULTS.md §4](NEGATIVE-RESULTS.md).

## 5. What would still be worth doing

A hand-written weight-stationary MXFP4 GEMV for M ≤ 8, one CTA per
(expert, N-tile), `cp.async.cg` 16 B loads, dequant in registers via a `prmt`
LUT, `ue8m0` decoded as `bits << 23`, compiled for `sm_121a` (the arch target
where NVIDIA's `cvt.rn.f16x2.e2m1x2` is actually accepted). Estimated 1–3
weeks and genuinely risky; the payoff is the 76 % → ~95 % step, i.e. roughly
+10 tok/s, plus a cleaner path for the 252 support kernels.

Before starting it: measure DRAM latency on GB10 directly and confirm with
`ncu` whether the kernel is bandwidth-bound or dequant/L1TEX-bound at these
shapes. The `sm_120a` data points in the literature suggest the latter is
possible, and it would change the design.
