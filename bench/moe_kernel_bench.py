"""Isolated benchmark of the MXFP4 MoE CUTLASS kernel at production shapes.

gpt-oss-120b: 128 experts, top-4, hidden 2880 -> padded 2944, expert
intermediate 2880 -> padded 2944, MXFP4 weights + MXFP8 activations.

Reports achieved bandwidth against the *actual* number of distinct experts
touched, which is what decides whether the kernel is at the roofline.
"""

import os
import sys

import torch

sys.path.insert(0, os.environ.get("FLASHINFER_SPARK_DIR", "/opt/flashinfer-spark"))
import flashinfer_spark.fused_moe as fm
from flashinfer_spark import block_scale_interleave, mxfp8_quantize

DEV = "cuda"
E = 128
TOPK = 4
H = 2944          # hidden, padded to 128
I = 2944          # expert intermediate, padded to 128
SF = 32           # mxfp4 block size
BW_PEAK = 273e9   # GB10 spec
BW_REAL = 250e9   # measured achievable (lab GEMV microbench)


def make_weights():
    w13 = torch.randint(0, 255, (E, 2 * I, H // 2), device=DEV, dtype=torch.uint8)
    w2 = torch.randint(0, 255, (E, H, I // 2), device=DEV, dtype=torch.uint8)
    w13_s = torch.randint(120, 132, (E, 2 * I, H // SF), device=DEV, dtype=torch.uint8)
    w2_s = torch.randint(120, 132, (E, H, I // SF), device=DEV, dtype=torch.uint8)
    # same interleave the serving path applies
    w13_s = block_scale_interleave(w13_s.view(torch.uint8)).reshape(w13_s.shape)
    w2_s = block_scale_interleave(w2_s.view(torch.uint8)).reshape(w2_s.shape)
    w13_b = torch.zeros(E, 2 * I, device=DEV, dtype=torch.bfloat16)
    w2_b = torch.zeros(E, H, device=DEV, dtype=torch.bfloat16)
    return w13, w2, w13_s, w2_s, w13_b, w2_b


def bytes_per_expert():
    w13 = 2 * I * (H // 2)
    w2 = H * (I // 2)
    s13 = 2 * I * (H // SF)
    s2 = H * (I // SF)
    return w13 + w2 + s13 + s2


def run(M, w, iters=50, warmup=10, distinct=None):
    w13, w2, w13_s, w2_s, w13_b, w2_b = w
    x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16) / 10
    xq, xs = mxfp8_quantize(x, True, 32)

    if distinct is None:
        # realistic: each token routes independently -> up to M*TOPK experts
        ids = torch.stack(
            [torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]
        ).to(torch.int32)
    else:
        # forced: all tokens share the same `distinct` experts
        base = torch.randperm(E, device=DEV)[:distinct]
        ids = base.repeat(M, 1)[:, :TOPK].contiguous().to(torch.int32)
    n_distinct = int(torch.unique(ids).numel())
    scales = torch.ones(M, TOPK, device=DEV, dtype=torch.float32) / TOPK

    alpha = torch.full((E,), 1.702, device=DEV, dtype=torch.float32)
    beta = torch.full((E,), 1.0, device=DEV, dtype=torch.float32)
    limit = torch.full((E,), 7.0, device=DEV, dtype=torch.float32)
    fake = torch.ones(E, device=DEV, dtype=torch.float32)
    qs = [w13_s.contiguous().view(torch.int32), fake,
          w2_s.contiguous().view(torch.int32), fake]
    out = torch.empty(M, H, device=DEV, dtype=torch.bfloat16)

    def call():
        fm.cutlass_fused_moe(
            xq, ids, scales,
            w13.contiguous().view(torch.long), w2.contiguous().view(torch.long),
            torch.bfloat16,
            quant_scales=qs,
            fc1_expert_biases=w13_b, fc2_expert_biases=w2_b,
            swiglu_alpha=alpha, swiglu_beta=beta, swiglu_limit=limit,
            input_sf=xs, use_mxfp8_act_scaling=True, output=out,
            tune_max_num_tokens=8,
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        call()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    traffic = n_distinct * bytes_per_expert()
    gbs = traffic / (ms / 1000) / 1e9
    print(
        f"M={M:<2} distinct experts={n_distinct:<3} {ms*1000:8.1f} us   "
        f"traffic {traffic/1e6:7.1f} MB   {gbs:6.1f} GB/s   "
        f"= {gbs/ (BW_REAL/1e9) *100:5.1f}% des erreichbaren / "
        f"{gbs/(BW_PEAK/1e9)*100:5.1f}% peak"
    )
    return ms


def sweep():
    """Sweep the SM120 tile shapes at production shapes, M=2 (our verify size).

    /tmp/flashinfer_moe_tile is re-read on every call, but each new tile shape
    triggers a JIT build of the kernel module (minutes on first use)."""
    import os
    import traceback

    w = make_weights()
    tiles = ["64x128", "128x128", "64x64", "64x32", "32x128", "128x64", "64x16"]
    print(f"per-expert bytes: {bytes_per_expert()/1e6:.2f} MB\n")
    for t in tiles:
        with open("/tmp/flashinfer_moe_tile", "w") as f:
            f.write(t)
        print(f"[tile {t}]", flush=True)
        try:
            run(2, w, iters=30, warmup=5)
        except Exception as e:
            print(f"   FEHLER: {type(e).__name__}: {str(e)[:150]}", flush=True)
        finally:
            torch.cuda.synchronize()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "m2":
        run(2, make_weights(), iters=30, warmup=8)
        return
    print(f"per-expert bytes: {bytes_per_expert()/1e6:.2f} MB "
          f"(w13+w2+scales, MXFP4)\n")
    w = make_weights()
    print("--- realistische Routung (jedes Token eigene top-4) ---")
    for M in (1, 2, 3, 4, 8):
        run(M, w)
    print("\n--- erzwungen: alle Tokens auf dieselben 4 Experten ---")
    for M in (1, 2, 4, 8):
        run(M, w, distinct=TOPK)


if __name__ == "__main__":
    main()
