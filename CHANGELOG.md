# Changelog

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
