"""Inference API + recording page.

    uvicorn app.main:app --reload          (from the project root)

POST /predict takes an audio file (the page sends 16 kHz mono WAV) and returns
the emotion probabilities and the log-mel spectrogram the CNN actually saw.
`analyse` and `spectrogram_png` are also used by the Gradio front end (app.py).
"""
import base64
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audio import HOP_LENGTH, SR, preprocess_waveform, sliding_windows  # noqa: E402
from src.models import Ensemble, load_checkpoint  # noqa: E402

# Every checkpoint in app/models/ is loaded; several are averaged as an ensemble.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "app" / "models"))
MAX_SECONDS = 15

torch.set_num_threads(max(1, os.cpu_count() // 2))
members = [load_checkpoint(p) for p in sorted(MODEL_DIR.glob("*.pt"))]
if not members:
    raise FileNotFoundError(f"no .pt checkpoint in {MODEL_DIR}")
model = members[0] if len(members) == 1 else Ensemble(members).eval()
CLASSES = members[0].classes
MODEL_NAME = members[0].name if len(members) == 1 else f"ensemble of {len(members)} CNNs"

app = FastAPI(title="Speech emotion recognition")


def spectrogram_png(wav: np.ndarray) -> bytes:
    with torch.no_grad():
        m = model.frontend.log_mel(torch.from_numpy(wav)[None])[0, 0].numpy()
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=110)
    ax.imshow(m, origin="lower", aspect="auto", cmap="magma",
              extent=[0, m.shape[1] * HOP_LENGTH / SR, 0, m.shape[0]])
    ax.set_xlabel("time (s)"); ax.set_ylabel("mel bin")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def analyse(wav: np.ndarray, sr: int):
    """Raw audio -> (probabilities sorted high to low, preprocessed waveform, n windows).
    Raises ValueError on unusable input."""
    if len(wav) / sr > MAX_SECONDS:
        wav = wav[: int(MAX_SECONDS * sr)]
    if len(wav) < sr * 0.3:
        raise ValueError("recording too short (less than 0.3 s)")
    wav = preprocess_waveform(wav, sr)
    x = sliding_windows(wav)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).mean(0).numpy()
    order = np.argsort(-probs)
    return {CLASSES[i]: float(probs[i]) for i in order}, wav, len(x)


def warm_up():
    """The first call compiles librosa's numba code and builds matplotlib's font
    cache (~25 s on CPU). Pay that at startup instead of on the first visitor."""
    rng = np.random.default_rng(0)
    wav = preprocess_waveform(rng.standard_normal(44100).astype(np.float32) * 0.1, 44100)
    with torch.no_grad():
        model(sliding_windows(wav))
    spectrogram_png(wav)


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "checkpoints": [m.name for m in members], "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        wav, sr = sf.read(io.BytesIO(await file.read()), dtype="float32")
    except Exception:
        raise HTTPException(400, "could not read audio, send a WAV/FLAC/OGG file")
    try:
        probs, wav, n_windows = analyse(wav, sr)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "prediction": next(iter(probs)),
        "probabilities": probs,
        "duration": len(wav) / SR,
        "n_windows": n_windows,
        "spectrogram_png": base64.b64encode(spectrogram_png(wav)).decode(),
        "model": MODEL_NAME,
    }


warm_up()
