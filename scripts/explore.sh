#!/usr/bin/env bash
# Round 2: ideas to improve on the baselines (seed 0 each). Selection is made on
# validation UAR only; the best settings are then re-run with 3 seeds.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
run() {
  local log="logs/$(echo "$*" | tr -d ' -.')_s0.log"
  .venv/bin/python -u -m src.train "$@" > "$log" 2>&1
  echo "$*: $(grep -h 'best epoch' "$log") | $(grep -h TEST "$log")"
}
run --arch cnn14 --epochs 40                        # audio-pretrained transfer (AudioSet)
run --arch cnn14 --freeze --epochs 20               # AudioSet features + linear head
run --arch resnet18 --upsample 2                    # ImageNet transfer on a 2x larger input
run --arch scratch --epochs 100                     # best epoch was ~58/60: train longer
run --arch scratch --mixup 0.4                      # mixup regularisation
run --arch scratch --noise                          # noise robustness (also helps the app)
run --arch scratch --width 64                       # 4x more parameters
run --arch cnn14 --epochs 40 --mixup 0.4 --noise    # best transfer + augmentations
