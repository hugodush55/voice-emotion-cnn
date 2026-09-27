#!/usr/bin/env bash
# Runs of scripts/confirm.sh that were interrupted when the session ended.
set -u
cd "$(dirname "$0")/.."
run() {
  local log="logs/$(echo "$*" | tr -d ' -.').log"
  .venv/bin/python -u -m src.train "$@" > "$log" 2>&1
  echo "$*: $(grep -h 'best epoch' "$log") | $(grep -h TEST "$log")"
}
run --arch scratch --width 64 --noise --seed 2
for s in 1 2; do run --arch cnn14 --epochs 40 --seed $s; done
