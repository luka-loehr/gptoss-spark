import json, re, sys

ref = json.load(open(sys.argv[1]))
cand = json.load(open(sys.argv[2]))

CHECKS = {
    4: [r"1[.,\s']?848[.,\s']?648"],
    7: [r"26[.,]6|26[.,]7"],
    11: [r"3\s*\(\s*x\s*\+\s*2\s*\)|3x\s*\+\s*6"],
    15: [r"3[.,]47|3[.,]5\b"],
    18: [r"\(x\s*\+\s*2\)\s*\(x\s*\+\s*3\)"],
}

def grade(item):
    i = item["i"]
    if i not in CHECKS:
        return None
    text = (item.get("content") or "") + " " + (item.get("reasoning") or "")
    return all(re.search(p, text) for p in CHECKS[i])

agree_total = pos_total = 0
top5_hits = top5_total = 0
dlp_sum = dlp_n = 0.0
rows = []
for r, c in zip(ref, cand):
    rt = [t["t"] for t in r["tokens"]]
    ct = [t["t"] for t in c["tokens"]]
    n = min(len(rt), len(ct))
    div = n
    for k in range(n):
        if rt[k] != ct[k]:
            div = k
            break
    agree_total += div
    pos_total += max(len(rt), len(ct))
    for k in range(div):
        rlp = r["tokens"][k]["lp"]; clp = c["tokens"][k]["lp"]
        if rlp is not None and clp is not None:
            dlp_sum += abs(rlp - clp); dlp_n += 1
    lookahead = min(div + 8, n)
    for k in range(div, lookahead):
        top5_total += 1
        if rt[k] in (c["tokens"][k]["top"] or []):
            top5_hits += 1
    rows.append((r["i"], div, len(rt), len(ct), grade(r), grade(c)))

print(f"{'idx':>3} {'agree-prefix':>12} {'ref-len':>7} {'cand-len':>8} {'ref-ok':>6} {'cand-ok':>7}")
for i, div, rl, cl, gr, gc in rows:
    print(f"{i:>3} {div:>12} {rl:>7} {cl:>8} {str(gr):>6} {str(gc):>7}")
print()
print(f"token agreement (aligned prefix / max len): {agree_total}/{pos_total} = {100*agree_total/pos_total:.1f}%")
print(f"mean |dlogprob| on agreed tokens: {dlp_sum/max(dlp_n,1):.4f} (n={int(dlp_n)})")
print(f"ref greedy token in cand top-5 around divergence: {top5_hits}/{top5_total}")
gr_ok = sum(1 for *_, g, _c in rows if g)
gc_ok = sum(1 for *_, _g, c2 in rows if c2)
n_checked = sum(1 for *_, g, _c in rows if g is not None)
print(f"auto-graded checkable answers: ref {gr_ok}/{n_checked} correct, cand {gc_ok}/{n_checked} correct")
