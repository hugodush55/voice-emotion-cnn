"""Audio preprocessing shared by training and the web app.

Everything a clip goes through before reaching the CNN lives here, so that
the training pipeline and the inference API are guaranteed to be identical.
"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio

SR = 16000             # CREMA-D is recorded at 16 kHz
CLIP_SECONDS = 3.0     # clips are 1.3-5 s long (mean 2.5 s)
CLIP_SAMPLES = int(SR * CLIP_SECONDS)
N_MELS = 64
N_FFT = 512
WIN_LENGTH = 400       # 25 ms
HOP_LENGTH = 160       # 10 ms -> 301 frames for 3 s
TRIM_TOP_DB = 30       # silence threshold for leading/trailing trim


def preprocess_waveform(wav: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 at 16 kHz, silence trimmed, peak-normalised.

    Peak normalisation removes the recording gain, which differs a lot between
    the corpus and a random laptop microphone.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    if len(trimmed) > SR // 10:  # keep the original if trimming removed nearly everything
        wav = trimmed
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.95
    return wav.astype(np.float32)


def fix_length(wav: np.ndarray, random: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop or zero-pad to CLIP_SAMPLES; centred at eval, random offset at train."""
    n = len(wav)
    if n >= CLIP_SAMPLES:
        start = rng.integers(0, n - CLIP_SAMPLES + 1) if random else (n - CLIP_SAMPLES) // 2
        return wav[start:start + CLIP_SAMPLES]
    out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    start = rng.integers(0, CLIP_SAMPLES - n + 1) if random else (CLIP_SAMPLES - n) // 2
    out[start:start + n] = wav
    return out


class LogMel(nn.Module):
    """Waveform batch (B, T) -> normalised log-mel spectrogram (B, 1, N_MELS, frames).

    The per-mel-bin mean/std are buffers: they are fitted on the training
    speakers only (see `fit`) and saved in the model checkpoint with it.
    """

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=20.0, f_max=SR / 2, power=2.0,
        )
        self.register_buffer("mean", torch.zeros(1, 1, N_MELS, 1))
        self.register_buffer("std", torch.ones(1, 1, N_MELS, 1))

    def log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.log(self.mel(wav) + 1e-6).unsqueeze(1)

    @torch.no_grad()
    def fit(self, wavs: torch.Tensor, batch_size: int = 256) -> None:
        """Compute per-bin statistics over a (N, T) tensor of training waveforms."""
        total, total_sq, count = 0.0, 0.0, 0
        for i in range(0, len(wavs), batch_size):
            m = self.log_mel(wavs[i:i + batch_size].to(self.mean.device))
            total = total + m.sum(dim=(0, 3), keepdim=True)
            total_sq = total_sq + (m ** 2).sum(dim=(0, 3), keepdim=True)
            count += m.shape[0] * m.shape[3]
        mean = total / count
        self.mean.copy_(mean)
        self.std.copy_((total_sq / count - mean ** 2).clamp_min(1e-8).sqrt())

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return (self.log_mel(wav) - self.mean) / self.std
