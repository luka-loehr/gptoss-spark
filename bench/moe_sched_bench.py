"""Pingpong vs Cooperative for the SM120 MXFP4 MoE GEMM at serving shapes.

The stock harness cannot force a union of more than TOPK experts (it slices the
pool back down to TOPK), and single 30-iteration timings on this box scatter by
+-5%, which is the same size as the effect being measured. So: build the routing
explicitly, and report the min of several independent repetitions.
"""

import os
import sys

sys.argv = ["x"]
sys.path.insert(0, "/opt/ffs")
sys.path.insert(0, "/bench")

with open("/tmp/flashinfer_moe_tile", "w") as f:
    f.write(os.environ["TILE"])

import torch  # noqa: E402
import moe_kernel_bench as mb  # noqa: E402
import flashinfer_spark.fused_moe as fm  # noqa: E402
from flashinfer_spark import mxfp8_quantize  # noqa: E402

DEV, E, TOPK, H = mb.DEV, mb.E, mb.TOPK, mb.H


def routing(M, distinct, gen):
    """M rows of TOPK expert ids whose union has exactly `distinct` members."""
    assert TOPK <= distinct <= M * TOPK
    pool = torch.randperm(E, generator=gen, device=DEV)[:distinct]
    rows, used = [], 0
    for m in range(M):
        fresh = min(distinct - used, TOPK) if m < M - 1 else distinct - used
        fresh = max(0, min(fresh, TOPK))
        take = pool[used : used + fresh].tolist()
        take += pool[: TOPK - len(take)].tolist()
        rows.append(take[:TOPK])
        used += fresh
    ids = torch.tensor(rows, device=DEV, dtype=torch.int32)
    assert int(torch.unique(ids).numel()) == distinct, (
        int(torch.unique(ids).numel()),
        distinct,
    )
    return ids


def bench(M, distinct, w, reps=5, iters=60, warmup=15):
    w13, w2, w13_s, w2_s, w13_b, w2_b = w
    gen = torch.Generator(device=DEV).manual_seed(1234)
    x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16, generator=gen) / 10
    xq, xs = mxfp8_quantize(x, True, 32)
    ids = routing(M, distinct, gen)
    scales = torch.ones(M, TOPK, device=DEV, dtype=torch.float32) / TOPK
    alpha = torch.full((E,), 1.702, device=DEV, dtype=torch.float32)
    beta = torch.full((E,), 1.0, device=DEV, dtype=torch.float32)
    limit = torch.full((E,), 7.0, device=DEV, dtype=torch.float32)
    fake = torch.ones(E, device=DEV, dtype=torch.float32)
    qs = [
        w13_s.contiguous().view(torch.int32),
        fake,
        w2_s.contiguous().view(torch.int32),
        fake,
    ]
    out = torch.empty(M, H, device=DEV, dtype=torch.bfloat16)
    w13c = w13.contiguous().view(torch.long)
    w2c = w2.contiguous().view(torch.long)

    def call():
        fm.cutlass_fused_moe(
            xq, ids, scales, w13c, w2c, torch.bfloat16,
            quant_scales=qs, fc1_expert_biases=w13_b, fc2_expert_biases=w2_b,
            swiglu_alpha=alpha, swiglu_beta=beta, swiglu_limit=limit,
            input_sf=xs, use_mxfp8_act_scaling=True, output=out,
            tune_max_num_tokens=8,
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    best = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            call()
        e.record()
        torch.cuda.synchronize()
        best.append(s.elapsed_time(e) / iters * 1000.0)
    best.sort()
    us = best[0]
    traffic = distinct * mb.bytes_per_expert()
    gbs = traffic / (us / 1e6) / 1e9
    print(
        "M=%d distinct=%-2d  min %7.1f us  med %7.1f us  %6.1f GB/s  %5.1f%% of achievable"
        % (M, distinct, us, best[len(best) // 2], gbs, gbs / 250.0 * 100),
        flush=True,
    )
    return us


w = mb.make_weights()
print(
    "[tile=%s cooperative=%s]"
    % (os.environ["TILE"], os.environ.get("FLASHINFER_MOE_COOPERATIVE", "0")),
    flush=True,
)
for M, d in ((1, 4), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)):
    bench(M, d, w)
