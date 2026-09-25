| run | seeds | trainable params | val UAR % | test acc % | test UAR % | test macro-F1 % | train min |
|---|---|---|---|---|---|---|---|
| resnet18_frozen_speakersplit | 1 | 3,078 | 41.1 | 39.6 | 39.7 | 37.5 | 1.9 |
| resnet18_speakersplit | 3 | 11,179,590 | 68.0 ± 0.1 | 66.4 ± 1.4 | 66.7 ± 1.4 | 66.3 ± 1.5 | 5.7 |
| scratch_randomsplit | 3 | 1,175,718 | 71.1 ± 0.3 | 69.9 ± 0.2 | 69.9 ± 0.3 | 69.9 ± 0.2 | 9.9 |
| scratch_speakersplit | 3 | 1,175,718 | 70.0 ± 0.8 | 68.7 ± 0.1 | 68.8 ± 0.2 | 68.6 ± 0.1 | 8.7 |
