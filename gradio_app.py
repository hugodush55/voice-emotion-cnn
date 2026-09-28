"""Gradio front end, used by the Hugging Face Space (and runnable locally):

    python gradio_app.py        -> http://localhost:7860

Same models and preprocessing as the FastAPI app (app/main.py). Record or upload
audio and get: the emotion probabilities of the selected model, the log-mel
spectrogram the CNN sees with a Grad-CAM overlay, the waveform with pitch and the
estimated glottal pulses (inverse filtering), and every model's answer side by side.
"""
import gradio as gr
import numpy as np
import torch

from app.main import CLASSES, DEFAULT_MODEL, MODEL_INFO, MODELS, analyse
from app.plots import comparison_figure, signal_figure, spectrogram_figure
from src.audio import SR, sliding_windows
from src.gradcam import grad_cam, supports_gradcam
from src.voice_analysis import glottal_flow, most_voiced_segment, pitch_track

try:  # Hugging Face ZeroGPU hardware requires at least one @spaces.GPU function
    import spaces
    on_zerogpu = spaces.GPU(duration=20)
except ImportError:  # running locally
    def on_zerogpu(fn):
        return fn


def classified_window(model, wav: np.ndarray, top: int) -> torch.Tensor:
    """The 3 s window to explain: the only one for short clips, otherwise the
    window most confident about the overall prediction."""
    wins = sliding_windows(wav)
    if len(wins) > 1:
        with torch.no_grad():
            wins = wins[int(torch.softmax(model(wins), 1)[:, top].argmax())][None]
    return wins


@on_zerogpu
def predict(audio, model_name=DEFAULT_MODEL):
    if audio is None:
        raise gr.Error("Record or upload some audio first.")
    sr, wav = audio
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    try:
        results, processed, n_windows = analyse(wav.astype(np.float32), sr)
    except ValueError as e:
        raise gr.Error(str(e))
    model_name = model_name if model_name in MODELS else DEFAULT_MODEL
    model, probs = MODELS[model_name], results[model_name]
    label = next(iter(probs))

    window = classified_window(model, processed, CLASSES.index(label))
    if supports_gradcam(model):
        cam, spec, _ = grad_cam(model, window, CLASSES.index(label))
    else:
        cam, spec = None, model.frontend.log_mel(window)[0, 0].detach().numpy()
    spec_img = spectrogram_figure(spec, cam, label)

    f0, f0_times = pitch_track(processed, SR)
    flow, dflow = glottal_flow(processed, SR)
    signal_img = signal_figure(processed, f0, f0_times, flow, dflow, most_voiced_segment(processed, SR, f0))

    compare_img = comparison_figure(results, MODEL_INFO, CLASSES, model_name)
    table = [[n, next(iter(r)), round(100 * next(iter(r.values()))), round(100 * MODEL_INFO[n]["test_uar"], 1),
              MODEL_INFO[n]["note"]] for n, r in results.items()]
    votes = [next(iter(r)) for r in results.values()]
    agree = max(set(votes), key=votes.count)
    info = (f"{len(processed) / SR:.2f} s after silence trimming, {n_windows} window"
            f"{'s' if n_windows > 1 else ''} of 3 s. Model: {model_name}. "
            f"{votes.count(agree)} of {len(votes)} models answer '{agree}'.")
    return probs, spec_img, signal_img, compare_img, table, info


def warm_up():
    """Compile pyin's numba code at startup rather than on the first visitor."""
    noise = (np.random.default_rng(0).standard_normal(SR) * 0.1).astype(np.float32)
    pitch_track(noise, SR)
    glottal_flow(noise, SR)


DESCRIPTION = """
Record a short sentence (2-4 s) and a CNN guesses the emotion: **anger, disgust, fear, happy, neutral or sad**.
Try saying *"It's eleven o'clock"* angrily, then sadly.

Trained on CREMA-D (91 actors) and evaluated on speakers never seen in training. The corpus contains
acted emotions in English, so predictions on spontaneous speech are much less reliable.
"""

with gr.Blocks(title="Voice Emotion CNN") as demo:
    gr.Markdown("# What does your voice sound like?")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your voice")
            model_choice = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model",
                                       info="All models are compared in the last tab; this one drives the "
                                            "prediction and the Grad-CAM.")
            button = gr.Button("Analyse", variant="primary")
            info = gr.Textbox(label="Details", interactive=False)
        with gr.Column(scale=1):
            label = gr.Label(num_top_classes=6, label="Predicted emotion")
    with gr.Tab("Spectrogram"):
        spec_img = gr.Image(label="What the CNN sees, and where it looked", type="pil")
    with gr.Tab("Waveform, pitch and vocal-fold pulses"):
        signal_img = gr.Image(label="Signal view", type="pil")
        gr.Markdown(
            "The lower panel is an **estimate** of the airflow through the vocal folds, obtained by "
            "inverse filtering (IAIF): linear prediction models the vocal tract and removes it from the "
            "speech. A real laryngogram needs electrodes on the neck; this one is computed from the audio only."
        )
    with gr.Tab("Compare models"):
        compare_img = gr.Image(label="Every model on your recording", type="pil")
        compare_table = gr.Dataframe(headers=["model", "prediction", "confidence %", "test UAR %", "about"],
                                     label="Same numbers as a table", interactive=False, wrap=True)
        gr.Markdown(
            "The test UAR is measured on 13 CREMA-D actors that no model heard during training "
            "(chance: 17 %). The first four (the ensemble and its three members) are trained from scratch; the last three start from "
            "networks pretrained on other data (transfer learning)."
        )
    outputs = [label, spec_img, signal_img, compare_img, compare_table, info]
    inputs = [audio, model_choice]
    button.click(predict, inputs=inputs, outputs=outputs, api_name="predict")
    audio.stop_recording(predict, inputs=inputs, outputs=outputs)
    model_choice.change(lambda a, m: predict(a, m) if a is not None else [gr.skip()] * len(outputs), inputs=inputs, outputs=outputs)

warm_up()

if __name__ == "__main__":
    demo.launch()
