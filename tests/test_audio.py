import numpy as np
import torch

from src.audio import CLIP_SAMPLES, N_MELS, SR, LogMel, fix_length, preprocess_waveform, sliding_windows


def test_fix_length_pads_and_crops_centred():
    short = np.ones(SR, dtype=np.float32)
    out = fix_length(short)
    assert len(out) == CLIP_SAMPLES and out.sum() == SR
    assert out[(CLIP_SAMPLES - SR) // 2] == 1 and out[0] == 0
    long = np.arange(2 * CLIP_SAMPLES, dtype=np.float32)
    assert fix_length(long)[0] == CLIP_SAMPLES // 2


def test_fix_length_random_stays_in_bounds():
    rng = np.random.default_rng(0)
    for n in [100, CLIP_SAMPLES, 3 * CLIP_SAMPLES]:
        assert len(fix_length(np.ones(n, np.float32), random=True, rng=rng)) == CLIP_SAMPLES


def test_preprocess_makes_mono_16k_trimmed_and_normalised():
    sr = 44100
    t = np.arange(sr) / sr
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([np.r_[np.zeros(sr), tone, np.zeros(sr)]] * 2, axis=1)  # 1 s silence each side
    out = preprocess_waveform(stereo, sr)
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) / SR - 1.0) < 0.15  # silence trimmed, resampled
    assert np.isclose(np.abs(out).max(), 0.95, atol=1e-3)


def test_sliding_windows():
    assert sliding_windows(np.zeros(SR, np.float32)).shape == (1, CLIP_SAMPLES)
    assert sliding_windows(np.zeros(10 * SR, np.float32)).shape == (8, CLIP_SAMPLES)  # 3 s windows, 1 s hop


def test_logmel_shape_and_fitted_normalisation():
    mel = LogMel()
    wavs = torch.randn(8, CLIP_SAMPLES) * torch.linspace(0.1, 1, 8)[:, None]
    mel.fit(wavs)
    out = mel(wavs)
    assert out.shape == (8, 1, N_MELS, 301)
    assert abs(out.mean()) < 0.05 and abs(out.std() - 1) < 0.05
