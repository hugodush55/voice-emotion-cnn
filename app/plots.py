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
    axes[0, 0].set_title("Log-mel spectrogram: the 3 s window the CNN receives", fontsize=10, loc="left")
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
