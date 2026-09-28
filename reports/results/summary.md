| run | runs (seeds/folds) | trainable params | val UAR % | test acc % | test UAR % | test macro-F1 % | train min |
|---|---|---|---|---|---|---|---|
| cnn14_ep40_speakersplit | 3 | 79,686,086 | 69.0 ± 0.2 | 68.7 ± 0.8 | 69.1 ± 0.8 | 68.4 ± 1.0 | 13.9 |
| cnn14_frozen_ep20_speakersplit | 1 | 12,294 | 43.3 | 45.4 | 45.2 | 44.5 | 2.3 |
| cnn14_mixup0.4_noise_ep40_speakersplit | 1 | 79,686,086 | 67.4 | 68.0 | 68.4 | 67.8 | 12.9 |
| dilated_w48_cv5split | 5 | 1,313,526 | 69.4 ± 2.7 | 66.6 ± 2.0 | 66.9 ± 2.0 | 66.8 ± 1.9 | 6.7 |
| dilated_w48_speakersplit | 3 | 1,313,526 | 71.6 ± 0.1 | 68.6 ± 0.4 | 68.8 ± 0.5 | 68.8 ± 0.5 | 8.8 |
| resnet18_frozen_speakersplit | 1 | 3,078 | 41.1 | 39.6 | 39.7 | 37.5 | 1.9 |
| resnet18_speakersplit | 3 | 11,179,590 | 68.0 ± 0.1 | 66.4 ± 1.4 | 66.7 ± 1.4 | 66.3 ± 1.5 | 5.7 |
| resnet18_up2_speakersplit | 1 | 11,179,590 | 69.2 | 67.4 | 67.7 | 67.3 | 11.3 |
| scratch_cv5split | 5 | 1,175,718 | 68.5 ± 3.4 | 66.0 ± 1.7 | 66.2 ± 1.6 | 65.9 ± 1.9 | 5.2 |
| scratch_ep100_speakersplit | 1 | 1,175,718 | 69.8 | 67.4 | 67.7 | 67.5 | 9.4 |
| scratch_mixup0.4_speakersplit | 1 | 1,175,718 | 69.7 | 68.8 | 69.0 | 68.7 | 5.8 |
| scratch_noise_speakersplit | 1 | 1,175,718 | 70.3 | 69.8 | 69.9 | 70.0 | 6.0 |
| scratch_randomsplit | 3 | 1,175,718 | 71.1 ± 0.3 | 69.9 ± 0.2 | 69.9 ± 0.3 | 69.9 ± 0.2 | 9.9 |
| scratch_speakersplit | 3 | 1,175,718 | 70.0 ± 0.8 | 68.7 ± 0.1 | 68.8 ± 0.2 | 68.6 ± 0.1 | 8.7 |
| scratch_w64_noise_speakersplit | 3 | 4,692,294 | 71.1 ± 0.4 | 68.4 ± 0.8 | 68.5 ± 0.8 | 68.5 ± 0.9 | 14.4 |
| scratch_w64_speakersplit | 1 | 4,692,294 | 71.5 | 69.4 | 69.5 | 69.7 | 12.4 |
