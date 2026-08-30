# Speculative decoding on a MoE model

## 1. Why the usual intuition fails here

On a dense model, verifying K+1 tokens costs about the same as decoding one:
the weights are read once either way. On gpt-oss-120b it does not, because
each token selects its own top-4 of 128 experts. Two tokens touch ~8 distinct
experts, so verification reads ~2× the expert bytes — measured directly in
[KERNELS.md §2](KERNELS.md).

Speculation therefore has to pay for its verification out of a much smaller
margin than the literature's dense-model numbers suggest.

## 2. The K-sweep

Measured with the NVIDIA `gpt-oss-120b-Eagle3-v3` head, drafter fully
MXFP4-quantized, `TRITON_ATTN`, full CUDA graphs:

| K | acceptance | tokens/pass | pass | throughput |
| ---: | ---: | ---: | ---: | ---: |
| — (no speculation) | — | 1.00 | 16.0 ms | 62.4 tok/s |
| **1** | 61 % | 1.61 | **24.7 ms** | **65.2 tok/s** |
| 2 | 50 % overall | 1.96 | 32.3 ms | 60.7 tok/s |

One draft step costs ~7 ms, of which only ~1.9 ms is the draft model's own
layer work. At 61 % acceptance the first step pays for itself; the second does
not. K=2 additionally drops GPU utilisation to 89.5 % (K=1 holds ~100 %),
i.e. the second step leaves the CUDA-graph path — an upstream issue worth
fixing for anyone who needs deeper trees.

## 3. Two upstream bugs had to be fixed first

Both are in [patches/04](../patches/04-gpt-oss-eagle3-aux.patch) and
[patches/05](../patches/05-eagle3-draft-quant.patch):

1. **Aux hidden-state boundary.** NVIDIA's head consumes hidden states from
   layers `[24, 30, 36]`. Index 36 means *after the last layer, pre-norm*, but
   the capture loop ends at 35 — so the drafter received 2 of 3 tensors and
   died on a shape mismatch (`5760` vs `8640`). The identical bug exists in
   SGLang 0.5.4, which is why Eagle3 never ran there either.
2. **Draft-model quantization.** `get_draft_quant_config` raises
   *"hf_overrides must be a dict"* for a local-path draft model and can return
   `None`; either way the drafter keeps bf16 weights. With a 201k-row bf16
   `lm_head` that is 1.16 GB per draft step.

## 4. Quantizing the drafter is worth 4 tok/s

`bench/draft_microbench.py` measures each draft layer against its bandwidth
floor. The `lm_head` via Marlin was already fine (1.33 ms, 93 % efficiency);
the **bf16 MLP** was not (1.49 ms for gate_up + down). Since gpt-oss itself is
MoE and has no dense MLP, matching `.gate_up_proj`/`.down_proj`/`.fc` can only
hit the drafter — safe by construction. Result: K=2 56.8 → 60.7, K=1 62.0 →
65.2, acceptance unchanged.

## 5. Retraining the head: a validated pipeline that did not pay

The theory was sound — NVIDIA trained the head against the *bf16* model, ours
is MXFP4 in both experts and dense layers, so a head trained on our model's
own outputs should accept more. The pipeline works and is reusable:

1. Capture hook in the target model dumps `(input_ids, positions, aux[N,8640])`
   per prefill chunk, env-gated, compile-safe when off (patches/04).
2. Labels come free from the API: `prompt_logprobs=1` yields the target's
   argmax per position — exactly what the verifier compares against.
3. `openai_harmony` renders self-generated chats into exact token IDs, which
   the driver submits as `prompt` IDs (no special-token drift).
4. A standalone trainer reproduces vLLM's `llama_eagle3` forward exactly —
   verified by the pretrained head reaching the same agreement in the trainer
   as in serving (32.34 % with vLLM's own rotary embedding vs 32.39 % with a
   hand-written YaRN; the match is what validates the reimplementation).

Outcome: offline agreement on serving-distribution data 61.0 % → 67.7 %,
serving acceptance and throughput unchanged. The measured reason is that
position-1 acceptance (the drafter's *own* recursion) did not improve —
single-step training on target hidden states cannot teach that. Multi-step
training plus roughly 10× the data is the honest next attempt; it was not
made because the expected value was below the alternatives.
