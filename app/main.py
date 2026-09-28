"""Inference API + recording page.

    uvicorn app.main:app --reload          (from the project root)

POST /predict takes an audio file (the page sends 16 kHz mono WAV) and returns
the emotion probabilities and the log-mel spectrogram the CNN actually saw.
`analyse` and `spectrogram_png` are also used by the Gradio front end (gradio_app.py).

The models offered are listed in app/models/registry.json; the first one is the
default. An entry with several checkpoints is an ensemble. Entries whose
checkpoints are missing are skipped (CNN14, 319 MB, is not in the git repository).
"""
import json
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

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "app" / "models"))
MAX_SECONDS = 15

torch.set_num_threads(max(1, os.cpu_count() // 2))
CHECKPOINTS = {}  # checkpoint name -> model, each loaded once even if shared by several entries
MODELS = {}       # display name -> model (a single CNN or an Ensemble)
MODEL_INFO = {}   # display name -> registry entry
for entry in json.loads((MODEL_DIR / "registry.json").read_text()):
    paths = [MODEL_DIR / f"{c}.pt" for c in entry["checkpoints"]]
    if not all(p.exists() for p in paths):
        print(f"skipping '{entry['name']}': checkpoint not found in {MODEL_DIR}")
        continue
    for p in paths:
        if p.stem not in CHECKPOINTS:
            CHECKPOINTS[p.stem] = load_checkpoint(p)
    ms = [CHECKPOINTS[p.stem] for p in paths]
    MODELS[entry["name"]] = ms[0] if len(ms) == 1 else Ensemble(ms).eval()
    MODEL_INFO[entry["name"]] = entry
if not MODELS:
    raise FileNotFoundError(f"no usable model in {MODEL_DIR}")
DEFAULT_MODEL = next(iter(MODELS))
model = MODELS[DEFAULT_MODEL]
CLASSES = next(iter(CHECKPOINTS.values())).classes

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
    """Raw audio -> ({model name: probabilities sorted high to low}, preprocessed
    waveform, n windows), for every available model. Each checkpoint runs once;
    an ensemble's probabilities are the mean of its members'. Raises ValueError
    on unusable input."""
    if len(wav) / sr > MAX_SECONDS:
        wav = wav[: int(MAX_SECONDS * sr)]
    if len(wav) < sr * 0.3:
        raise ValueError("recording too short (less than 0.3 s)")
    wav = preprocess_waveform(wav, sr)
    x = sliding_windows(wav)
    with torch.no_grad():
        per_ckpt = {name: torch.softmax(m(x), dim=1).mean(0).numpy() for name, m in CHECKPOINTS.items()}
    results = {}
    for name, entry in MODEL_INFO.items():
        probs = np.mean([per_ckpt[c] for c in entry["checkpoints"]], axis=0)
        results[name] = {CLASSES[i]: float(probs[i]) for i in np.argsort(-probs)}
    return results, wav, len(x)


def warm_up():
    """The first call compiles librosa's numba code and builds matplotlib's font
    cache (~25 s on CPU). Pay that at startup instead of on the first visitor."""
    rng = np.random.default_rng(0)
    wav = preprocess_waveform(rng.standard_normal(44100).astype(np.float32) * 0.1, 44100)
    analyse(wav, SR)
    spectrogram_png(wav)


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL, "models": {n: e["checkpoints"] for n, e in MODEL_INFO.items()},
            "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        wav, sr = sf.read(io.BytesIO(await file.read()), dtype="float32")
    except Exception:
        raise HTTPException(400, "could not read audio, send a WAV/FLAC/OGG file")
    try:
        results, wav, n_windows = analyse(wav, sr)
    except ValueError as e:
        raise HTTPException(400, str(e))
    probs = results[DEFAULT_MODEL]
    return {
        "prediction": next(iter(probs)),
        "probabilities": probs,
        "duration": len(wav) / SR,
        "n_windows": n_windows,
        "spectrogram_png": base64.b64encode(spectrogram_png(wav)).decode(),
        "model": DEFAULT_MODEL,
        "all_models": results,
    }


warm_up()
