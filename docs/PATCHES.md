# The patches

Five diffs, 471 lines total, against vLLM `0.28.1rc1.dev43+g6f7df92a8`. Three
of them work around genuine upstream bugs; two are Spark-specific enablement.
Apply with `ops/apply-patches.sh` or let `containers/Dockerfile` do it.

## 01 — MoE backend selection (16 lines)

`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`

Upstream already contains the full gpt-oss MXFP4×MXFP8 CUTLASS path, including
`sm_12x` device checks — but `_get_requested_backends` filters it out when the
model does not request an activation format, which gpt-oss does not. The patch
adds the same special case that `b12x` already has: on an explicit
`--moe-backend flashinfer_cutlass`, return the MXFP8-activation variant first.

Without this, `--moe-backend flashinfer_cutlass` fails with
*"does not support the deployment configuration since kernel does not support
quantization scheme QuantKey(u8,scale(u8,static,GroupShape(row=1, col=32)),symmetric)"*.

## 02 — SM121 kernels (70 lines)

`vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`

Routes the fused-MoE call to `flashinfer_spark` (the vendored fork) when
`VLLM_SPARK_CUTLASS=1`. The engine's own FlashInfer stays in place — it is
needed for `sm_121` attention and is a *newer* release than the fork, so
downgrading it is not an option; the two coexist under different import names.

The shim filters keyword arguments against the fork's signature and remaps the
`ActivationType` enum, so the call site keeps working when either side moves.
Bundled FlashInfer 0.6.17 does have MXFP4 CUTLASS MoE, but on `sm_121` it is
~5× slower than the fork's kernels (34.7 vs 62.4 tok/s end to end).

## 03 — MXFP4 for dense layers (273 lines)

`vllm/model_executor/layers/quantization/mxfp4.py`

The largest win in the repository. gpt-oss ships MXFP4 *expert* weights;
`qkv_proj`, `o_proj` and the 201088×2880 `lm_head` remain bf16 in every stock
configuration. This adds two quantization methods that runtime-quantize those
weights to MXFP4 (packed e2m1 + ue8m0 block scales, linear scale layout) and
execute them through Marlin's fused dequant-GEMM.

Enabled per layer group via `VLLM_SPARK_DENSE=qkv,o,lm_head,mlp,fc`. The
`mlp`/`fc` entries only ever match the Eagle3 drafter — gpt-oss itself is MoE
and has no dense MLP — which is what makes them safe to match by name.

Ported from the same fork the kernels come from, adapted to current upstream:
`apply_fp4_marlin_linear` renamed its third argument to `weight_global_scale`,
and the lm_head path needs the structural tied-embedding check
(`data_ptr` comparison) because `tie_word_embeddings` is not always readable
at that point.

## 04 — Eagle3 aux hidden states (63 lines)

`vllm/model_executor/models/gpt_oss.py`

Two changes:

1. **The bug.** NVIDIA's `gpt-oss-120b-Eagle3-v3` head consumes hidden states
   from layers `[24, 30, 36]`. Index 36 means "after the last layer,
   pre-norm", but the capture loop runs `range(start_layer, end_layer)` and
   stops at 35. The drafter then receives 2 of 3 tensors and dies with
   `mat1 and mat2 shapes cannot be multiplied (Nx5760 and 8640x2880)`.
   The patch appends the boundary state when the id equals `end_layer`.
   *The identical bug exists in SGLang 0.5.4* — it is why Eagle3 never ran
   there either.
2. **A capture hook** for retraining the draft head: when
   `VLLM_SPARK_CAPTURE_DIR` is set, each prefill chunk dumps
   `(input_ids, positions, aux[N,8640])`. Module-level constant, so
   `torch.compile` folds it away when unset; capture runs need
   `--enforce-eager`. See [SPECULATION.md §5](SPECULATION.md).

Also passes `vllm_config.quant_config` to `ParallelLMHead`, without which
patch 03 cannot reach the target's lm_head.

## 05 — Eagle3 draft quantization (49 lines)

`vllm/model_executor/models/llama_eagle3.py`

`get_draft_quant_config` raises *"hf_overrides must be a dict"* for a
local-path draft model, and can also return `None`. Either way the drafter
keeps bf16 weights — with a 201k-row bf16 `lm_head` that is 1.16 GB of extra
traffic *per draft step*. The patch falls back to the target's quantization
config.

Worth 13.7 tok/s on its own in the speculative profile (43.1 → 56.8 when it
first landed on the older stack).

## Refreshing against a newer nightly

```bash
docker run --rm --entrypoint bash vllm/vllm-openai:nightly -c \
  'cat /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py' > new.py
patch -p1 --dry-run < patches/03-mxfp4-dense-layers.patch   # check for drift
```

Patches 01, 04 and 05 fix upstream bugs and should be sent upstream rather
than carried forever — 04 in particular affects every Eagle3 user of gpt-oss,
not just this hardware.
