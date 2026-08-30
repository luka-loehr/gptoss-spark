"""Decompose the Eagle draft step: time every layer at M=1 (decode) and M=3
(verify batch), against its memory-traffic floor, and compare candidate
implementations for the expensive ones.

Run inside the vllm container with a GPU.
"""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)

DEV = "cuda"
DT = torch.bfloat16
H = 2880
VOCAB = 201088
INTER = 16384
AUX = 8640
QKV_IN, QKV_OUT = 2 * H, 4096 + 512 + 512
O_IN = 4096
BW = 250e9  # achievable GB10 bandwidth, B/s


def mxfp4_quantize(x):
    from flashinfer import fp4_quantize

    mx = torch.clamp(x.abs().max().float(), min=1e-30)
    gs = ((448 * 6) / mx).to(device=x.device, dtype=torch.float32)
    q, s = fp4_quantize(x, gs, sf_vec_size=32, sf_use_ue8m0=True,
                        is_sf_swizzled_layout=False)
    if s.ndim == 1:
        s = s.view(x.size(0), -1)
    return q, s


class MarlinLinear(torch.nn.Module):
    """Mirrors SparkMxfp4LinearMethod: runtime MXFP4 + Marlin dequant-GEMM."""

    def __init__(self, out_features, in_features):
        super().__init__()
        w = (torch.randn(out_features, in_features, device=DEV, dtype=DT) / 20)
        self.output_size_per_partition = out_features
        self.input_size_per_partition = in_features
        self.params_dtype = DT
        q, s = mxfp4_quantize(w)
        self.weight = torch.nn.Parameter(q, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(s, requires_grad=False)
        prepare_fp4_layer_for_marlin(self, input_dtype=DT)

    def forward(self, x):
        return apply_fp4_marlin_linear(
            input=x,
            weight=self.weight,
            weight_scale=self.weight_scale,
            weight_global_scale=None,
            workspace=self.workspace,
            size_n=self.output_size_per_partition,
            size_k=self.input_size_per_partition,
            bias=None,
        )


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def report(name, ms, bytes_moved, note=""):
    floor = bytes_moved / BW * 1000
    eff = floor / ms * 100 if ms > 0 else 0
    print(
        f"{name:<34} {ms:7.3f} ms   floor {floor:6.3f} ms   "
        f"eff {eff:5.1f}%   {note}"
    )
    return ms


def main():
    torch.manual_seed(0)
    print(f"{'component':<34} {'measured':>10}   {'bandwidth floor':>15}   eff")
    print("-" * 92)

    for M in (1, 3):
        x_h = torch.randn(M, H, device=DEV, dtype=DT)
        x_aux = torch.randn(M, AUX, device=DEV, dtype=DT)
        x_qkv = torch.randn(M, QKV_IN, device=DEV, dtype=DT)
        x_o = torch.randn(M, O_IN, device=DEV, dtype=DT)
        x_int = torch.randn(M, INTER, device=DEV, dtype=DT)
        print(f"\n=== M = {M} ===")
        total = 0.0

        # fc: 8640 -> 2880, bf16 in the checkpoint (not covered by our matcher)
        fc = torch.nn.Linear(AUX, H, bias=False, device=DEV, dtype=DT)
        total += report("fc  bf16 (8640->2880)", bench(lambda: fc(x_aux)),
                        AUX * H * 2)
        fc_q = MarlinLinear(H, AUX)
        report("fc  MXFP4/Marlin", bench(lambda: fc_q(x_aux)), AUX * H * 0.5,
               "candidate")

        # attention projections (already MXFP4 in our setup)
        qkv = MarlinLinear(QKV_OUT, QKV_IN)
        total += report("qkv MXFP4/Marlin", bench(lambda: qkv(x_qkv)),
                        QKV_IN * QKV_OUT * 0.5)
        o = MarlinLinear(H, O_IN)
        total += report("o   MXFP4/Marlin", bench(lambda: o(x_o)), O_IN * H * 0.5)

        # MLP: bf16 today (our matcher does not cover gate_up/down)
        gate_up = torch.nn.Linear(H, 2 * INTER, bias=False, device=DEV, dtype=DT)
        down = torch.nn.Linear(INTER, H, bias=False, device=DEV, dtype=DT)
        total += report("mlp gate_up bf16", bench(lambda: gate_up(x_h)),
                        H * 2 * INTER * 2)
        total += report("mlp down    bf16", bench(lambda: down(x_int)),
                        INTER * H * 2)
        gate_up_q = MarlinLinear(2 * INTER, H)
        down_q = MarlinLinear(H, INTER)
        report("mlp gate_up MXFP4/Marlin", bench(lambda: gate_up_q(x_h)),
               H * 2 * INTER * 0.5, "candidate")
        report("mlp down    MXFP4/Marlin", bench(lambda: down_q(x_int)),
               INTER * H * 0.5, "candidate")

        # lm_head: the big one
        lm_q = MarlinLinear(VOCAB, H)
        t_lm = report("lm_head MXFP4/Marlin", bench(lambda: lm_q(x_h), iters=100),
                      H * VOCAB * 0.5 + VOCAB * (H // 32))
        total += t_lm
        lm_bf16 = torch.nn.Linear(H, VOCAB, bias=False, device=DEV, dtype=DT)
        report("lm_head bf16 (reference)", bench(lambda: lm_bf16(x_h), iters=100),
               H * VOCAB * 2, "for comparison")

        # sampling / argmax over the vocab
        logits = torch.randn(M, VOCAB, device=DEV, dtype=torch.float32)
        total += report("argmax over vocab", bench(lambda: logits.argmax(-1)),
                        M * VOCAB * 4)
        report("softmax+top-k(50)", bench(lambda: torch.topk(logits, 50, -1)),
               M * VOCAB * 4, "if sampling")

        print(f"{'SUM of draft-step components':<34} {total:7.3f} ms")


if __name__ == "__main__":
    main()
