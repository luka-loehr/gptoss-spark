#!/usr/bin/env python3
"""Cut the Eagle3 draft head's lm_head down to the tokens it actually proposes.

NVIDIA's gpt-oss-120b Eagle3 heads carry the full 201088-row vocabulary and no
d2t mapping, so every draft step reads a 307 MB MXFP4 lm_head -- 1.33 ms of a
24.6 ms decode pass, 27 % of all dense weight traffic, to produce one throwaway
proposal. Restricting it to the N tokens the target model actually emits keeps
the same head with an N-row lm_head plus the d2t offset table vLLM already
supports (`Eagle3LlamaForCausalLM.compute_logits`).

This cannot change what the model outputs. The verifier still runs the target's
full lm_head; only the *proposals* come from a smaller list, so a token outside
it costs one rejected draft, never a different answer.

Measured at N=32768 on serving-distribution data: held-out coverage 98.3 %,
acceptance 61 % -> 59.2 %, decode 65.4 -> 65.9 tok/s. A modest win on its own;
it composes with the kernel patches.

Usage:
    # 1. collect target-model token frequencies (any JSON list of token ids per
    #    request works; --labels may be given more than once)
    # 2. build the head
    python3 ops/shrink-draft-vocab.py \
        --head /models/gpt-oss-120b-Eagle3-v3 \
        --out  /models/gpt-oss-120b-Eagle3-v3-vocab32k \
        --labels /captures/round1 --labels /captures/round2 \
        --size 32768
"""

import argparse
import collections
import glob
import json
import os
import shutil


def token_counts(dirs):
    """Frequency of target-emitted token ids across every JSON file in `dirs`.

    Each file is a list of token ids (negative entries are masked positions).
    """
    counts = collections.Counter()
    files = 0
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                ids = json.load(open(path))
            except (ValueError, OSError):
                continue
            if not isinstance(ids, list):
                continue
            files += 1
            for tok in ids:
                if isinstance(tok, int) and tok >= 0:
                    counts[tok] += 1
    return counts, files


def build_shortlist(counts, size, vocab_size, special_from):
    """`size` target ids: every special token, then by frequency, then by id.

    Special tokens are unconditional -- the drafter has to be able to propose
    <|return|> and friends, and they are frequent in chat traffic. The tail is
    filled by ascending id, which for a BPE vocabulary is a merge-frequency
    prior and covers tokens the capture happened not to contain.
    """
    keep = set(range(special_from, vocab_size))
    if len(keep) > size:
        raise SystemExit("size %d is smaller than the %d special tokens" % (size, len(keep)))
    for tok, _ in counts.most_common():
        if len(keep) >= size:
            break
        keep.add(tok)
    tok = 0
    while len(keep) < size:
        if tok not in keep:
            keep.add(tok)
        tok += 1
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="source Eagle3 head directory")
    ap.add_argument("--out", required=True, help="destination directory")
    ap.add_argument("--labels", action="append", default=[],
                    help="directory of captured target token-id JSON files (repeatable)")
    ap.add_argument("--size", type=int, default=32768, help="draft vocabulary size")
    ap.add_argument("--special-from", type=int, default=200000,
                    help="first id treated as a special token (o200k_harmony: 200000)")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    cfg = json.load(open(os.path.join(args.head, "config.json")))
    vocab_size = cfg["vocab_size"]

    counts, n_files = token_counts(args.labels)
    print("read %d capture files, %d distinct token ids" % (n_files, len(counts)))
    keep = build_shortlist(counts, args.size, vocab_size, args.special_from)
    from_data = sum(1 for t in keep if t < args.special_from and counts[t] > 0)
    print("shortlist: %d ids (%d special, %d seen in captures, %d id-prior fill)"
          % (len(keep), vocab_size - args.special_from, from_data,
             len(keep) - (vocab_size - args.special_from) - from_data))

    shards = sorted(glob.glob(os.path.join(args.head, "*.safetensors")))
    if len(shards) != 1:
        raise SystemExit("expected exactly one safetensors shard, found %d" % len(shards))
    sd = load_file(shards[0])
    if "lm_head.weight" not in sd:
        raise SystemExit("head has no lm_head.weight (it ties to the target's); nothing to shrink")

    idx = torch.tensor(keep, dtype=torch.long)
    lm = sd["lm_head.weight"]
    print("lm_head %s -> (%d, %d)" % (tuple(lm.shape), len(keep), lm.shape[1]))
    sd["lm_head.weight"] = lm.index_select(0, idx).contiguous()
    # vLLM reconstructs target ids as arange(draft_vocab) + d2t.
    sd["d2t"] = (idx - torch.arange(len(keep), dtype=torch.long)).contiguous()

    os.makedirs(args.out, exist_ok=True)
    save_file(sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    cfg["draft_vocab_size"] = len(keep)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump(keep, open(os.path.join(args.out, "keep_ids.json"), "w"))
    for extra in ("README.md",):
        src = os.path.join(args.head, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, extra))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
