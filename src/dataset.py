"""CREMA-D indexing, speaker-independent splits and PyTorch dataset.

Run `python -m src.dataset` to build the waveform cache, write the split file
and save example spectrograms to reports/figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.audio import SR, fix_length, preprocess_waveform

ROOT = Path(__file__).resolve().parents[1]
CREMA_DIR = ROOT / "data" / "CREMA-D"
CACHE_PATH = ROOT / "data" / "cache" / "crema_waves.npz"
SPLIT_PATH = ROOT / "data" / "splits.csv"

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
EMOTION_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
LABEL = {e: i for i, e in enumerate(EMOTIONS)}


def build_index(crema_dir: Path = CREMA_DIR) -> pd.DataFrame:
    """One row per clip. Filenames look like 1001_DFA_ANG_XX.wav
    (actor _ sentence _ emotion _ intensity)."""
    rows = []
    for path in sorted((crema_dir / "AudioWAV").glob("*.wav")):
        actor, sentence, emotion, intensity = path.stem.split("_")
        rows.append(dict(path=str(path), actor=int(actor), sentence=sentence,
                         emotion=emotion, intensity=intensity, label=LABEL[emotion]))
    df = pd.DataFrame(rows)
    demo = pd.read_csv(crema_dir / "VideoDemographics.csv")
    df = df.merge(demo[["ActorID", "Sex", "Age"]].rename(columns={"ActorID": "actor", "Sex": "sex", "Age": "age"}),
                  on="actor", how="left")
    return df


def speaker_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                  seed: int = 0) -> pd.DataFrame:
    """Assign every *actor* to exactly one of train/val/test, balanced by sex."""
    rng = np.random.default_rng(seed)
    split_of = {}
    for _, actors in df.groupby("sex")["actor"]:
        actors = rng.permutation(actors.unique())
        n_test, n_val = round(len(actors) * test_frac), round(len(actors) * val_frac)
        for i, a in enumerate(actors):
            split_of[a] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df = df.assign(split=df["actor"].map(split_of))
    check_speaker_disjoint(df)
    return df


def cv_speaker_split(df: pd.DataFrame, fold: int, n_folds: int = 5, val_frac: float = 0.15,
                     seed: int = 0) -> pd.DataFrame:
    """Leave-speakers-out cross-validation: the 91 actors are dealt into
    `n_folds` sex-balanced groups; fold k is the test set, and a sex-balanced
    `val_frac` of the actors is held out from the remaining folds for model
    selection. Over the folds, every actor is a test speaker exactly once."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, actors in df.groupby("sex")["actor"]:
        for i, a in enumerate(rng.permutation(actors.unique())):
            fold_of[a] = i % n_folds
    split_of = {a: "test" for a, f in fold_of.items() if f == fold}
    rest = df[~df.actor.isin(split_of)]
    for _, actors in rest.groupby("sex")["actor"]:
        actors = rng.permutation(actors.unique())
        n_val = round(len(actors) * val_frac)
        split_of.update({a: "val" if i < n_val else "train" for i, a in enumerate(actors)})
    df = df.assign(split=df["actor"].map(split_of), fold=df["actor"].map(fold_of))
    check_speaker_disjoint(df)
    return df


def random_clip_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                      seed: int = 0) -> pd.DataFrame:
    """The WRONG way (clips shuffled regardless of speaker). Only used for the
    leakage experiment in the report."""
    rng = np.random.default_rng(seed)
    u = rng.random(len(df))
    split = np.where(u < test_frac, "test", np.where(u < test_frac + val_frac, "val", "train"))
    return df.assign(split=split)


def check_speaker_disjoint(df: pd.DataFrame) -> None:
    spk = {s: set(g["actor"]) for s, g in df.groupby("split")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = spk.get(a, set()) & spk.get(b, set())
        assert not overlap, f"speakers {sorted(overlap)} appear in both {a} and {b}"


def load_waveform_cache(df: pd.DataFrame, cache_path: Path = CACHE_PATH) -> list[np.ndarray]:
    """Preprocessed variable-length waveforms, in the order of df['path'].
    Stored once as int16 (about 570 MB) so training never touches the WAV files."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        names, flat, offsets = data["names"], data["flat"], data["offsets"]
        by_name = {n: flat[offsets[i]:offsets[i + 1]] for i, n in enumerate(names)}
        missing = [p for p in df["path"] if Path(p).name not in by_name]
        if not missing:
            return [by_name[Path(p).name].astype(np.float32) / 32767 for p in df["path"]]
        print(f"cache is missing {len(missing)} clips, rebuilding")

    full = build_index()
    waves = []
    for p in tqdm(full["path"], desc="preprocessing"):
        wav, sr = sf.read(p, dtype="float32")
        waves.append(preprocess_waveform(wav, sr))
    offsets = np.concatenate([[0], np.cumsum([len(w) for w in waves])])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, names=np.array([Path(p).name for p in full["path"]]),
             flat=(np.concatenate(waves) * 32767).astype(np.int16), offsets=offsets)
    return load_waveform_cache(df, cache_path)


class SERDataset(Dataset):
    """Returns (fixed-length waveform, label). The log-mel is computed on the
    GPU inside the model (src.audio.LogMel), batch by batch."""

    def __init__(self, waves: list[np.ndarray], labels, train: bool, seed: int = 0):
        self.waves, self.labels, self.train = waves, np.asarray(labels), train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        wav = fix_length(self.waves[i], random=self.train, rng=self.rng)
        return torch.from_numpy(wav), int(self.labels[i])


def load_split(df: pd.DataFrame | None = None, split_path: Path = SPLIT_PATH) -> pd.DataFrame:
    """Index + the saved speaker split (created on first call)."""
    if df is None:
        df = build_index()
    if split_path.exists():
        split = pd.read_csv(split_path)
        df = df.merge(split, on="actor", how="left")
        check_speaker_disjoint(df)
        return df
    df = speaker_split(df)
    df[["actor", "split"]].drop_duplicates().sort_values("actor").to_csv(split_path, index=False)
    return df


def _plot_examples(df: pd.DataFrame, waves: list[np.ndarray], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.audio import LogMel

    mel = LogMel()
    ex = df[(df.actor == 1001) & (df.sentence == "IEO") & (df.intensity.isin(["HI", "XX"]))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, emo, name in zip(axes.flat, EMOTIONS, EMOTION_NAMES):
        i = ex.index[ex.emotion == emo][0]
        m = mel.log_mel(torch.from_numpy(fix_length(waves[i]))[None])[0, 0].numpy()
        ax.imshow(m, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(fix_length(waves[i])) / SR, 0, m.shape[0]])
        ax.set_title(f"{name}  ({Path(df.path[i]).name})", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mel bin")
    fig.suptitle("Actor 1001, sentence 'It's eleven o'clock' - log-mel spectrograms (before normalisation)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    df = load_split()
    waves = load_waveform_cache(df)

    print(f"{len(df)} clips, {df.actor.nunique()} actors\n")
    summary = df.groupby("split").agg(actors=("actor", "nunique"), clips=("path", "size"),
                                      male=("sex", lambda s: s.eq("Male").sum()))
    summary["male_%"] = (100 * summary.male / summary.clips).round(1)
    print(summary.drop(columns="male").loc[["train", "val", "test"]], "\n")
    print(pd.crosstab(df.split, df.emotion).loc[["train", "val", "test"]], "\n")
    for s in ["val", "test"]:
        print(f"{s} actors: {sorted(df.loc[df.split == s, 'actor'].unique().tolist())}")
    lens = np.array([len(w) for w in waves]) / SR
    print(f"\nafter silence trim: {lens.min():.2f}-{lens.max():.2f} s, mean {lens.mean():.2f} s, "
          f"{100 * (lens > 3).mean():.1f}% longer than 3 s")

    _plot_examples(df, waves, ROOT / "reports" / "figures" / "mel_examples.png")
    print("saved reports/figures/mel_examples.png")
