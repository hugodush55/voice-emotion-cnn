"""Gradio front end, used by the Hugging Face Space (and runnable locally):

    python gradio_app.py        -> http://localhost:7860

Same model and preprocessing as the FastAPI app (app/main.py): record or upload
audio, see the log-mel spectrogram the CNN receives, get the emotion probabilities.
"""
import io

import gradio as gr
import numpy as np
from PIL import Image

from app.main import analyse, ckpt, spectrogram_png

try:  # Hugging Face ZeroGPU hardware requires at least one @spaces.GPU function
    import spaces
    on_zerogpu = spaces.GPU(duration=20)
except ImportError:  # running locally
    def on_zerogpu(fn):
        return fn


@on_zerogpu
def predict(audio):
    if audio is None:
        raise gr.Error("Record or upload some audio first.")
    sr, wav = audio
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    try:
        probs, processed, n_windows = analyse(wav.astype(np.float32), sr)
    except ValueError as e:
        raise gr.Error(str(e))
    spectrogram = Image.open(io.BytesIO(spectrogram_png(processed)))
    info = (f"{len(processed) / 16000:.2f} s after silence trimming, "
            f"{n_windows} window{'s' if n_windows > 1 else ''} of 3 s, model: {ckpt['arch']}")
    return probs, spectrogram, info


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your voice (2-4 s)"),
    outputs=[
        gr.Label(num_top_classes=6, label="Predicted emotion"),
        gr.Image(label="Log-mel spectrogram (what the CNN sees)", type="pil"),
        gr.Textbox(label="Details"),
    ],
    title="What does your voice sound like?",
    description=(
        "Record a short sentence and a CNN guesses the emotion: anger, disgust, fear, happy, "
        "neutral or sad. Try saying *\"It's eleven o'clock\"* angrily, then sadly.\n\n"
        "Trained on CREMA-D (91 actors) and evaluated on speakers never seen in training. "
        "The corpus contains acted emotions in English, so predictions on spontaneous speech "
        "are much less reliable."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
