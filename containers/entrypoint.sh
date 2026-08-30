#!/usr/bin/env bash
# gptoss-spark entrypoint: pick a serving profile, then hand over to vLLM.
set -Eeuo pipefail

MODEL_DIR="${MODEL_DIR:-/model}"
EAGLE_DIR="${EAGLE_DIR:-/eagle}"
PORT="${PORT:-8100}"
SERVED_NAME="${SERVED_NAME:-gptoss}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
# spec  = single-user record (65 tok/s, 2 slots, Eagle3 K=1)
# plain = multi-user (62 tok/s single, 300 tok/s aggregate at 30 users)
PROFILE="${PROFILE:-plain}"

if [[ ! -r "${MODEL_DIR}/config.json" ]]; then
  echo "gptoss-spark: no checkpoint at ${MODEL_DIR}." >&2
  echo "Mount the gpt-oss-120b MXFP4 weights there, e.g.:" >&2
  echo "  -v /path/to/gpt-oss-120b:/model:ro" >&2
  exit 2
fi

common=(
  --served-model-name "${SERVED_NAME}"
  --host 0.0.0.0 --port "${PORT}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --enable-prefix-caching
  --moe-backend flashinfer_cutlass
)

case "${PROFILE}" in
  plain)
    exec vllm serve "${MODEL_DIR}" "${common[@]}" \
      --max-num-seqs "${MAX_NUM_SEQS:-32}" \
      --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8192}" \
      --load-format fastsafetensors "$@"
    ;;
  spec)
    if [[ ! -r "${EAGLE_DIR}/config.json" ]]; then
      echo "gptoss-spark: PROFILE=spec needs an Eagle3 head at ${EAGLE_DIR}." >&2
      echo "  -v /path/to/gpt-oss-120b-Eagle3-v3:/eagle:ro" >&2
      exit 2
    fi
    exec vllm serve "${MODEL_DIR}" "${common[@]}" \
      --max-num-seqs "${MAX_NUM_SEQS:-2}" \
      --attention-backend TRITON_ATTN \
      --speculative-config "{\"method\":\"eagle3\",\"model\":\"${EAGLE_DIR}\",\"num_speculative_tokens\":${SPEC_K:-1},\"quantization\":\"gpt_oss_mxfp4\"}" \
      "$@"
    ;;
  *)
    echo "gptoss-spark: PROFILE must be 'plain' or 'spec' (got '${PROFILE}')." >&2
    exit 2
    ;;
esac
