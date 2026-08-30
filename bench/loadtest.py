"""Concurrent load test: fires N streaming requests at once and reports the
true aggregate throughput (all tokens / makespan) plus TTFT distribution."""

import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
N = int(sys.argv[3])
MAX_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 512

PROMPTS = [
    "Explain how a modern operating system schedules processes. Be detailed.",
    "Erkläre ausführlich, wie Photosynthese funktioniert.",
    "Write a Python implementation of a priority queue with explanations.",
    "Beschreibe die Ursachen des Ersten Weltkriegs ausführlich.",
]

results = []
lock = threading.Lock()


def worker(i):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAX_TOK,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    toks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]" or not d:
                continue
            try:
                v = json.loads(d)
            except Exception:
                continue
            u = v.get("usage") or {}
            if u.get("completion_tokens"):
                usage = u["completion_tokens"]
            ch = (v.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning") or delta.get(
                "reasoning_content"
            ):
                if ttft is None:
                    ttft = time.time() - t0
                toks += 1
    end = time.time()
    with lock:
        results.append(
            {
                "ttft": ttft or (end - t0),
                "tokens": usage or toks,
                "start": t0,
                "end": end,
            }
        )


start_all = time.time()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
makespan = time.time() - start_all

total_tokens = sum(r["tokens"] for r in results)
ttfts = sorted(r["ttft"] for r in results)
per_user = [
    r["tokens"] / max(r["end"] - r["start"] - r["ttft"], 0.001) for r in results
]


def pct(xs, p):
    return xs[min(int(len(xs) * p), len(xs) - 1)]


print(f"users            : {N}")
print(f"makespan         : {makespan:.1f} s")
print(f"total tokens     : {total_tokens}")
print(f"AGGREGATE        : {total_tokens / makespan:.1f} tok/s")
print(f"per user (median): {statistics.median(per_user):.1f} tok/s")
print(f"TTFT median      : {statistics.median(ttfts):.2f} s")
print(f"TTFT p90 / max   : {pct(ttfts, 0.9):.2f} s / {max(ttfts):.2f} s")
