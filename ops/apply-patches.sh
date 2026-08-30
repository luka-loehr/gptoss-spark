#!/usr/bin/env bash
# Apply the patches to a *running or existing* vLLM install without rebuilding
# the image - handy while iterating on a patch.
#
#   TARGET=/usr/local/lib/python3.12/dist-packages ops/apply-patches.sh
#   ops/apply-patches.sh --reverse     # undo
set -Eeuo pipefail
TARGET="${TARGET:-/usr/local/lib/python3.12/dist-packages}"
extra=()
[[ "${1:-}" == "--reverse" ]] && extra+=(--reverse)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for p in "${here}"/patches/*.patch; do
  echo "==> $(basename "$p")"
  patch -p1 --forward "${extra[@]}" -d "${TARGET}" < "$p"
done
