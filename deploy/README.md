---
title: Voice Emotion CNN
emoji: 🎙️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Record your voice, see its spectrogram, get an emotion
---

# Voice Emotion CNN

Record a short sentence in the browser. The page shows the log-mel spectrogram
and the emotion predicted by a CNN (anger, disgust, fear, happy, neutral, sad).

The model is trained on CREMA-D (91 actors) with a speaker-independent split, so
the reported scores are measured on voices the network never heard during training.
The emotions in the corpus are acted, in English, so predictions on spontaneous
speech are much less reliable.

API: `POST /predict` with an audio file (`file` form field) returns the
probabilities and a base64 PNG of the spectrogram. `GET /health` names the model.
