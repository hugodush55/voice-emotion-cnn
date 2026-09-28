---
title: Voice Emotion CNN
emoji: 🎙️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: "3.10"
app_file: gradio_app.py
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

The free Space goes to sleep after 48 h without visitors: the first visit then
shows "Starting" for about a minute before the page appears.
