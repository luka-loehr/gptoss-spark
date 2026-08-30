# Changelog

## 0.2.0 — 2026-08-30

**Single stream 65.2 → 68.9 tok/s speculative, 62.4 → 64.3 plain** — both
measured from the published image, started against the real checkpoint with the
MoE module deleted from its JIT cache so it compiled that kernel from its own
patched source. Aggregate throughput at 30 concurrent users is unchanged within
run-to-run noise, which is what a *fixed* per-layer cost predicts: at high
concurrency there is enough real work per layer to absorb it.

- `patches/kernels/01-moe-sf-padding-loop.patch` — `expandInputRowsKernel` and
  `doActivationKernel` ended with a loop over `alignment × num_experts`
  scale-factor padding slots (16384 iterations for gpt-oss) that reloaded two
  `expert_first_token_offset` entries from global memory on every pass. At
  decode batch sizes only a handful of experts are routed to, so over 99 % of
  the iterations wrote nothing. Inverted to run once per expert; 12–15 µs
  cheaper per MoE call and bit-identical in output (verified by dumping both
  builds at fixed inputs).
- `ops/shrink-draft-vocab.py` — cuts an Eagle3 head's 201088-row `lm_head` to
  the tokens the target actually emits and emits the `d2t` table vLLM already
  supports. At 32768 rows: 98.3 % held-out coverage, acceptance 61 % → 59.2 %,
  +0.5 tok/s. Cannot change an answer — the target still verifies over the
  full vocabulary.
- `bench/moe_sched_bench.py` — forces an exact distinct-expert count (the older
  harness silently could not exceed `TOPK`) and reports the min of 5×60
  iterations, because single 30-iteration timings on this box scatter by ±5 %.

**Corrected**: the MoE expert GEMM does *not* stream at 76 % of achievable with
+10 tok/s left in it. Separating the fixed and marginal terms gives ≈59 µs per
layer + ≈62 µs per expert touched; the marginal rate is 223 GB/s, i.e. **89 %**.
The 76 % figure averaged the fixed cost into the per-expert rate. The
hand-written MXFP4 GEMV is correspondingly less attractive than documented.

**Corrected**: "smaller tile shapes crash with illegal instruction" was one bug
seen six times. The fork's `-DLOGICAL_TILE_M/N` flags rewrite the device-side
kernel while the host launcher keeps picking its tile from a four-shape
heuristic; only `64x128` survives, and only because at M ≤ 64 it yields the
same tile counts as `128x128`.

## 0.1.0 — 2026-08-30

First public release of the DGX Spark serving recipe for gpt-oss-120b.

**Measured** (one GB10, `sm_121`, identical harness for every number):

- single stream 65.2 tok/s speculative / 62.4 tok/s plain, up from 52.6 tok/s
  on the previously deployed SGLang stack
- 299.5 tok/s aggregate at 30 concurrent users, up from 244.1
- TTFT under 30-user load 1.2 s, down from 16.4 s
- 83 GiB memory cap, down from 96.3 GiB
- answer-level quality parity against the previous stack on a 20-prompt eval

**Contents**: five patches against vLLM `0.28.1rc1.dev43+g6f7df92a8`, a
container recipe that vendors the SM121 CUTLASS MXFP4 kernels under a separate
import name, the benchmark harness used for every number, the raw measurement
artifacts, and the record of fifteen approaches that did not work.

Known limits: patches are diffs against one nightly and need refreshing;
the kernels come from a personal upstream fork pinned by commit; tool-call
parity and a domain quality gate are open before this replaces a production
deployment (see `docs/SERVING.md §5`).
