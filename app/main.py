"""Inference API + recording page.

    uvicorn app.main:app --reload          (from the project root)

POST /predict takes an audio file (the page sends 16 kHz mono WAV) and returns
the emotion probabilities and the log-mel spectrogram the CNN actually saw.
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
from src.audio import CLIP_SAMPLES, HOP_LENGTH, SR, fix_length, preprocess_waveform  # noqa: E402
from src.models import SERModel  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "app" / "model.pt"))
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "checkpoints" / "scratch_speakersplit_s0.pt"
MAX_SECONDS = 15
WINDOW_HOP = SR  # 1 s hop between 3 s windows for long recordings

torch.set_num_threads(max(1, os.cpu_count() // 2))
ckpt = torch.load(MODEL_PATH, map_location="cpu")
kwargs = {**ckpt["model_kwargs"], "pretrained": False} if ckpt["arch"] == "resnet18" else {}
model = SERModel(ckpt["arch"], len(ckpt["classes"]), **kwargs)
model.load_state_dict(ckpt["state_dict"])
model.eval()
CLASSES = ckpt["classes"]

app = FastAPI(title="Speech emotion recognition")


def windows(wav: np.ndarray) -> torch.Tensor:
    """One centred 3 s window for short clips, 3 s windows every 1 s for long ones."""
    if len(wav) <= CLIP_SAMPLES:
        return torch.from_numpy(fix_length(wav))[None]
    starts = range(0, len(wav) - CLIP_SAMPLES + 1, WINDOW_HOP)
    return torch.from_numpy(np.stack([wav[s:s + CLIP_SAMPLES] for s in starts]))


def spectrogram_png(wav: np.ndarray) -> str:
    with torch.no_grad():
        m = model.frontend.log_mel(torch.from_numpy(wav)[None])[0, 0].numpy()
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=110)
    ax.imshow(m, origin="lower", aspect="auto", cmap="magma",
              extent=[0, m.shape[1] * HOP_LENGTH / SR, 0, m.shape[0]])
    ax.set_xlabel("time (s)"); ax.set_ylabel("mel bin")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


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
    if len(wav) / sr > MAX_SECONDS:
        wav = wav[: int(MAX_SECONDS * sr)]
    if len(wav) < sr * 0.3:
        raise HTTPException(400, "recording too short")
    wav = preprocess_waveform(wav, sr)
    x = windows(wav)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).mean(0).numpy()
    order = np.argsort(-probs)
    return {
        "prediction": CLASSES[order[0]],
        "probabilities": {CLASSES[i]: float(probs[i]) for i in order},
        "duration": len(wav) / SR,
        "n_windows": len(x),
        "spectrogram_png": spectrogram_png(wav),
        "model": ckpt["arch"],
    }
