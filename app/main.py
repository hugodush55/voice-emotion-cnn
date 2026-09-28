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
from src.models import SERModel  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "app" / "model.pt"))
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "checkpoints" / "scratch_speakersplit_s0.pt"
MAX_SECONDS = 15

torch.set_num_threads(max(1, os.cpu_count() // 2))
ckpt = torch.load(MODEL_PATH, map_location="cpu")
# pretrained weights are already inside the checkpoint, no need to download them again
kwargs = {**ckpt["model_kwargs"], **({"pretrained": False} if "pretrained" in ckpt["model_kwargs"] else {})}
model = SERModel(ckpt["arch"], len(ckpt["classes"]), **kwargs)
model.load_state_dict(ckpt["state_dict"])
model.eval()
CLASSES = ckpt["classes"]

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
    return {"status": "ok", "model": ckpt["arch"], "checkpoint": MODEL_PATH.name,
            "val_uar": ckpt.get("val_uar"), "classes": CLASSES}


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
        "model": ckpt["arch"],
    }


warm_up()
