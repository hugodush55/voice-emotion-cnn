#!/usr/bin/env bash
# Round 4: the course's time-dilated CNN, and leave-speakers-out 5-fold CV so
# that every one of the 91 actors is scored as an unseen test speaker once.
set -u
cd "$(dirname "$0")/.."
run() {
  local log="logs/$(echo "$*" | tr -d ' -.').log"
  .venv/bin/python -u -m src.train "$@" > "$log" 2>&1
  echo "$*: $(grep -h 'best epoch' "$log") | $(grep -h TEST "$log")"
}
for s in 0 1 2; do run --arch dilated --width 48 --seed $s; done
for f in 0 1 2 3 4; do run --arch scratch --split cv --fold $f; done
for f in 0 1 2 3 4; do run --arch dilated --width 48 --split cv --fold $f; done
