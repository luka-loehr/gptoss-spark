#!/usr/bin/env bash
# Build the serving image on an sm_121 host and push it to GHCR.
#
#   OWNER=your-gh-user VERSION=0.1.0 ops/publish-ghcr.sh
#
# Needs: docker login ghcr.io (a PAT with write:packages), and this repo as CWD.
set -Eeuo pipefail

OWNER="${OWNER:?set OWNER to your GitHub user/org}"
VERSION="${VERSION:?set VERSION, e.g. 0.1.0}"
IMAGE="ghcr.io/${OWNER}/gptoss-spark"

arch="$(uname -m)"
[[ "$arch" == "aarch64" || "$arch" == "arm64" ]] || {
  echo "publish-ghcr: build on the arm64 target host (got ${arch})." >&2; exit 2; }

echo "==> building ${IMAGE}:${VERSION}"
docker build -f containers/Dockerfile -t "${IMAGE}:${VERSION}" -t "${IMAGE}:latest" .

echo "==> smoke test (entrypoint refuses to start without a checkpoint)"
docker run --rm "${IMAGE}:${VERSION}" 2>&1 | grep -q "no checkpoint" \
  && echo "    ok: entrypoint guards a missing checkpoint"

echo "==> pushing"
docker push "${IMAGE}:${VERSION}"
docker push "${IMAGE}:latest"
echo "==> done: docker pull ${IMAGE}:${VERSION}"
