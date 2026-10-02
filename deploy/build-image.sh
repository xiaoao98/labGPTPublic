#!/usr/bin/env bash
# Build the container image for the lab server and hand it over, either as a file or
# through the internal registry.
#
#   ./deploy/build-image.sh                      build only
#   ./deploy/build-image.sh save                 build, then write labgpt-<tag>.tar.gz
#   ./deploy/build-image.sh push hpcharbor...    build, then push to a registry
#
# linux/amd64 is explicit because this is usually built on an arm64 laptop, where the
# default produces an image that loads on the server and then will not start.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${LABGPT_TAG:-$(date +%Y%m%d)-$(git rev-parse --short HEAD)}"
IMAGE="labgpt:$TAG"
MODE="${1:-build}"

echo "==> building $IMAGE for linux/amd64"
docker build --platform linux/amd64 -f deploy/Dockerfile -t "$IMAGE" -t labgpt:latest .

case "$MODE" in
  build)
    docker images "$IMAGE" --format "built {{.Repository}}:{{.Tag}}  {{.Size}}"
    ;;
  save)
    OUT="${2:-$HOME/Desktop/labgpt-$TAG.tar.gz}"
    echo "==> saving to $OUT"
    docker save "$IMAGE" | gzip > "$OUT"
    echo "saved $OUT ($(du -h "$OUT" | cut -f1))"
    echo "  on the server: gunzip -c labgpt-$TAG.tar.gz | docker load"
    ;;
  push)
    REGISTRY="${2:?usage: build-image.sh push <registry>/<project>}"
    echo "==> pushing to $REGISTRY/$IMAGE"
    docker tag "$IMAGE" "$REGISTRY/$IMAGE"
    docker push "$REGISTRY/$IMAGE"
    echo "  on the server: docker pull $REGISTRY/$IMAGE"
    ;;
  *)
    echo "unknown mode: $MODE (use build, save or push)" >&2; exit 1 ;;
esac

cat <<'NOTE'

The image carries no corpus. Mount the model and the index when running it:

  docker run -d --name labgpt --restart unless-stopped -p 8090:8090 \
      -v /path/to/bge-small-en-v1.5:/models/bge-small-en-v1.5:ro \
      -v /path/to/index:/index:ro \
      -v /path/to/data:/data \
      --env-file /path/to/labgpt.env \
      labgpt:TAG
NOTE
