"""Inverse filtering is checked on a synthetic vowel whose glottal flow is known."""
import numpy as np
import pytest
from scipy.signal import lfilter

from src.voice_analysis import glottal_flow, pitch_track

SR, F0 = 16000, 120.0


@pytest.fixture(scope="module")
def synthetic_vowel():
    """Rosenberg glottal pulses -> 3-formant vocal tract (/a/-like) -> lip radiation."""
    t = np.arange(int(0.6 * SR))
    phase = (t % (SR / F0)) / (SR / F0)
    t_open, t_close = 0.4, 0.16
    flow = np.where(phase < t_open, 0.5 * (1 - np.cos(np.pi * phase / t_open)),
                    np.where(phase < t_open + t_close, np.cos(np.pi * (phase - t_open) / (2 * t_close)), 0.0))
    a = np.array([1.0])
    for freq, bw in [(700, 80), (1220, 90), (2600, 120)]:
        r = np.exp(-np.pi * bw / SR)
        a = np.convolve(a, [1, -2 * r * np.cos(2 * np.pi * freq / SR), r * r])
    speech = lfilter([1, -1], [1], lfilter([1], a, flow))
    return speech / np.abs(speech).max(), flow


def test_iaif_recovers_the_glottal_pulse(synthetic_vowel):
    speech, true = synthetic_vowel
    est, _ = glottal_flow(speech, SR)
    seg = slice(int(0.2 * SR), int(0.4 * SR))  # steady part
    true, est = true[seg] - true[seg].mean(), est[seg] - est[seg].mean()
    best = max(np.corrcoef(true, np.roll(est, k))[0, 1] for k in range(-40, 41))
    # a pure 120 Hz sine scores ~0.75 against this pulse, so require clearly more
    assert best > 0.9


def test_pitch_track(synthetic_vowel):
    f0, times = pitch_track(synthetic_vowel[0], SR)
    assert len(f0) == len(times)
    assert abs(np.nanmedian(f0) - F0) < 2
