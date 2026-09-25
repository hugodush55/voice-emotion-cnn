#!/usr/bin/env bash
# Assemble the Hugging Face Space in build/space and upload it.
#   scripts/deploy_space.sh               build + upload to $SPACE
#   scripts/deploy_space.sh --build-only  build only
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE="${SPACE:-hugodush/voice-emotion-cnn}"
OUT=build/space

rm -rf "$OUT" && mkdir -p "$OUT/app/static" "$OUT/src"
cp deploy/Dockerfile deploy/requirements.txt deploy/README.md "$OUT/"
cp app/main.py app/model.pt "$OUT/app/"
cp app/static/index.html "$OUT/app/static/"
cp src/__init__.py src/audio.py src/models.py "$OUT/src/"
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"

[ "${1:-}" = "--build-only" ] && exit 0
"${HF:-hf}" repo create "$SPACE" --repo-type space --space-sdk docker --exist-ok
"${HF:-hf}" upload "$SPACE" "$OUT" . --repo-type space --commit-message "Deploy $(git rev-parse --short HEAD)"
echo "https://huggingface.co/spaces/$SPACE"
