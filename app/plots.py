"""Figures shown by the app: spectrogram + Grad-CAM, and waveform + pitch +
estimated glottal flow."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

from src.audio import HOP_LENGTH, SR

ACCENT = "#c2410c"


def _to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def spectrogram_figure(spec: np.ndarray, cam: np.ndarray | None, label: str) -> Image.Image:
    """spec: (n_mels, frames) log-mel of the 3 s window the CNN classified."""
    rows = 2 if cam is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(9, 2.6 * rows), sharex=True, squeeze=False)
    extent = [0, spec.shape[1] * HOP_LENGTH / SR, 0, spec.shape[0]]
    axes[0, 0].imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=extent)
    title = "Log-mel spectrogram: the 3 s window the CNN receives"
    if cam is None:
        title += " (Grad-CAM is shown for the from-scratch CNNs and their ensemble)"
    axes[0, 0].set_title(title, fontsize=10, loc="left")
    if cam is not None:
        axes[1, 0].imshow(spec, origin="lower", aspect="auto", cmap="gray", extent=extent)
        axes[1, 0].imshow(cam, origin="lower", aspect="auto", cmap="inferno", alpha=0.55, extent=extent,
                          vmin=0, vmax=1)
        axes[1, 0].set_title(f"Grad-CAM: regions that pushed the prediction towards '{label}'",
                             fontsize=10, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("mel band")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    return _to_image(fig)


def comparison_figure(results: dict, info: dict, classes: list, selected: str) -> Image.Image:
    """One row per model, one column per emotion, cell = probability (%). The
    row's top emotion is outlined; the model currently selected is marked."""
    names = list(results)
    probs = np.array([[results[n][c] for c in classes] for n in names])
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.6))
    ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        top = int(probs[i].argmax())
        for j in range(len(classes)):
            ax.text(j, i, f"{100 * probs[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if probs[i, j] > 0.55 else "#0b0b0b", fontweight="bold" if j == top else None)
        ax.add_patch(plt.Rectangle((top - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ACCENT, lw=2))
    labels = [f"{'> ' if n == selected else ''}{n}  ({100 * info[n]['test_uar']:.0f} %)" for n in names]
    ax.set_yticks(range(len(names)), labels, fontsize=9)
    ax.set_xticks(range(len(classes)), classes, fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Probability (%) given by each model; in brackets its test UAR on 13 unseen speakers",
                 fontsize=10, loc="left", pad=24)
    fig.tight_layout()
    return _to_image(fig)


def signal_figure(wav, f0, f0_times, flow, dflow, segment) -> Image.Image:
    """Waveform with the pitch contour, and a zoom on the estimated glottal pulses."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), gridspec_kw={"height_ratios": [1, 1.1]})
    t = np.arange(len(wav)) / SR
    ax1.plot(t, wav, color="0.6", lw=0.5)
    ax1.set_ylabel("amplitude"); ax1.set_xlabel("time (s)"); ax1.set_xlim(0, t[-1])
    s0, s1 = segment
    ax1.axvspan(s0 / SR, s1 / SR, color=ACCENT, alpha=0.15, lw=0)
    ax1b = ax1.twinx()
    ax1b.plot(f0_times, f0, color=ACCENT, lw=2)
    ax1b.set_ylabel("pitch F0 (Hz)", color=ACCENT)
    voiced = f0[np.isfinite(f0)]
    title = "Waveform and pitch contour"
    if len(voiced):
        title += f" (median F0 {np.median(voiced):.0f} Hz, range {voiced.min():.0f}-{voiced.max():.0f} Hz)"
        ax1b.set_ylim(max(0, voiced.min() * 0.8), voiced.max() * 1.2)
    ax1.set_title(title, fontsize=10, loc="left")

    tz = (np.arange(s0, s1) - s0) / SR * 1000
    zoom = wav[s0:s1] / (np.abs(wav[s0:s1]).max() + 1e-9)
    ax2.plot(tz, zoom, color="0.75", lw=1, label="speech")
    ax2.plot(tz, dflow[s0:s1] / (np.abs(dflow[s0:s1]).max() + 1e-9) * 0.8 - 2.3, color="0.3", lw=1,
             label="flow derivative")
    g = flow[s0:s1] / (np.abs(flow[s0:s1]).max() + 1e-9)
    ax2.plot(tz, g, color=ACCENT, lw=2, label="estimated glottal flow")
    # glottal closure instants = sharpest negative peaks of the flow derivative
    d = dflow[s0:s1]
    gci, _ = find_peaks(-d, distance=int(SR / 500), height=0.35 * np.abs(d).max())
    for k in gci:
        ax2.axvline(tz[k], color=ACCENT, lw=0.8, ls=":")
    ax2.set_yticks([]); ax2.set_ylim(-3.3, 1.9); ax2.set_xlabel("time (ms), shaded region above")
    ax2.set_title("Estimated laryngogram (inverse filtering, IAIF): vocal-fold pulses, "
                  "dotted = glottal closures", fontsize=10, loc="left")
    ax2.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    return _to_image(fig)
