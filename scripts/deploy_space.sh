#!/usr/bin/env bash
# Assemble the Hugging Face Space in build/space and push it with git over SSH.
#   scripts/deploy_space.sh               build + push to $SPACE
#   scripts/deploy_space.sh --build-only  build only
# One-time setup: SSH public key added at huggingface.co/settings/keys and the
# (empty, Gradio SDK) Space created at huggingface.co/new-space.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE="${SPACE:-hugodush/voice-emotion-cnn}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_personal}"
OUT=build/space
REPO=build/space-repo

rm -rf "$OUT" && mkdir -p "$OUT/app/static" "$OUT/src"
cp deploy/requirements.txt deploy/README.md gradio_app.py "$OUT/"
mkdir -p "$OUT/app/models"
cp app/main.py app/plots.py "$OUT/app/"
cp app/models/*.pt "$OUT/app/models/"
cp app/static/index.html "$OUT/app/static/"
cp src/__init__.py src/audio.py src/models.py src/gradcam.py src/voice_analysis.py "$OUT/src/"
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
[ "${1:-}" = "--build-only" ] && exit 0

export GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes"
export PATH="$HOME/.local/bin:$PATH"  # git-lfs: HF rejects binary files (model.pt) outside LFS
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull -q
else
  git clone -q "git@hf.co:spaces/$SPACE" "$REPO"
  git -C "$REPO" lfs install --local
  git -C "$REPO" config user.name "$(git config user.name)"
  git -C "$REPO" config user.email "$(git config user.email)"
fi
# replace everything except git metadata and the Space's .gitattributes
find "$REPO" -mindepth 1 -maxdepth 1 ! -name .git ! -name .gitattributes -exec rm -rf {} +
cp -r "$OUT"/. "$REPO"/
git -C "$REPO" add -A
if git -C "$REPO" diff --cached --quiet; then
  echo "nothing changed, Space already up to date"
else
  git -C "$REPO" commit -q -m "Deploy $(git rev-parse --short HEAD)"
  git -C "$REPO" push -q
  echo "pushed, the Space rebuilds in ~5 min: https://huggingface.co/spaces/$SPACE"
fi
