#!/usr/bin/env bash
# Round 3: re-run the settings that looked best on validation in round 2
# with 3 seeds, so differences can be compared with seed-to-seed noise.
set -u
cd "$(dirname "$0")/.."
run() {
  local log="logs/$(echo "$*" | tr -d ' -.').log"
  .venv/bin/python -u -m src.train "$@" > "$log" 2>&1
  echo "$*: $(grep -h 'best epoch' "$log") | $(grep -h TEST "$log")"
}
run --arch resnet18 --upsample 2 --seed 0           # crashed in round 2 (CUDA error)
for s in 0 1 2; do run --arch scratch --width 64 --noise --seed $s; done
for s in 1 2; do run --arch cnn14 --epochs 40 --seed $s; done   # seed 0 done in round 2
