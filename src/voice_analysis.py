"""Voice-source analysis for the app's signal view: pitch (F0) and an estimate
of the glottal flow, the 'estimated laryngogram' of the course slides.

Speech = vocal-fold pulses filtered by the vocal tract (+ lip radiation).
IAIF (Iterative Adaptive Inverse Filtering, Alku 1992) estimates the vocal
tract with LPC and removes it, leaving the airflow through the glottis. A real
laryngograph measures vocal-fold *contact* with neck electrodes, a different
physical signal: the pulse timing should match, the shape need not.
"""
import librosa
import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt

LEAK = 0.99  # leaky integrator, cancels the +6 dB/octave lip radiation


def pitch_track(wav: np.ndarray, sr: int, hop: int = 160):
    """F0 in Hz per 10 ms frame (NaN when unvoiced), and the frame times."""
    f0, voiced, _ = librosa.pyin(wav, fmin=65, fmax=500, sr=sr, frame_length=1024, hop_length=hop)
    f0 = np.where(voiced, f0, np.nan)
    return f0, librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)


def _lpc(x: np.ndarray, order: int) -> np.ndarray:
    x = x * np.hanning(len(x))
    if np.abs(x).max() < 1e-6:
        return np.r_[1.0, np.zeros(order)]
    try:
        a = librosa.lpc(x + 1e-9 * np.random.default_rng(0).standard_normal(len(x)), order=order)
    except FloatingPointError:
        return np.r_[1.0, np.zeros(order)]
    return a if np.all(np.isfinite(a)) else np.r_[1.0, np.zeros(order)]


def _integrate(x: np.ndarray) -> np.ndarray:
    return lfilter([1.0], [1.0, -LEAK], x)


def iaif_frame(x: np.ndarray, sr: int, vt_order: int | None = None, gl_order: int = 4) -> np.ndarray:
    """Glottal flow derivative of one short frame (IAIF, two iterations)."""
    p = vt_order or 2 + sr // 1000
    # 1st pass: remove the spectral tilt of the glottal pulse, then estimate the vocal tract
    y = lfilter(_lpc(x, 1), [1.0], x)
    g1 = _integrate(lfilter(_lpc(y, p), [1.0], x))
    # 2nd pass: better glottal-tilt model from g1, then a refined vocal-tract estimate
    y = _integrate(lfilter(_lpc(g1, gl_order), [1.0], x))
    return lfilter(_lpc(y, p), [1.0], x)


def glottal_flow(wav: np.ndarray, sr: int, frame_ms: float = 32, hop_ms: float = 16):
    """IAIF over the whole utterance (frame-wise, overlap-add).
    Returns (flow, flow_derivative), both normalised to [-1, 1]."""
    x = sosfiltfilt(butter(4, 60, "highpass", fs=sr, output="sos"), wav)  # remove rumble below F0
    n, h = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    win = np.hanning(n + 1)[:n]  # periodic Hann: 50 % overlap-add sums to 1
    dflow = np.zeros(len(x) + n)
    xp = np.pad(x, (0, n))
    for start in range(0, len(x), h):
        frame = xp[start:start + n]
        if np.abs(frame).max() > 1e-4:
            dflow[start:start + n] += iaif_frame(frame, sr) * win
    dflow = dflow[:len(x)]
    flow = sosfiltfilt(butter(2, 40, "highpass", fs=sr, output="sos"), _integrate(dflow))  # remove drift
    norm = lambda s: s / (np.abs(s).max() + 1e-9)
    return norm(flow), norm(dflow)


def most_voiced_segment(wav: np.ndarray, sr: int, f0: np.ndarray, hop: int = 160, ms: float = 60):
    """(start, end) sample indices of a `ms`-long window centred on the loudest voiced frame."""
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop)[0][:len(f0)]
    rms = np.where(np.isfinite(f0[:len(rms)]), rms, 0)
    centre = int(np.argmax(rms)) * hop if rms.max() > 0 else len(wav) // 2
    half = int(sr * ms / 2000)
    start = int(np.clip(centre - half, 0, max(0, len(wav) - 2 * half)))
    return start, min(len(wav), start + 2 * half)
