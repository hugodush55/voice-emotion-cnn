"""The evaluation is only honest if no speaker is ever in two splits."""
import numpy as np
import pandas as pd
import pytest

from src.dataset import CREMA_DIR, build_index, check_speaker_disjoint, cv_speaker_split, random_clip_split, \
    speaker_split


@pytest.fixture
def fake_corpus():
    """40 actors (22 male, 18 female) x 30 clips, same columns as build_index()."""
    rows = [dict(path=f"{a}_{i}.wav", actor=a, sex="Male" if a < 1022 else "Female", label=i % 6)
            for a in range(1000, 1040) for i in range(30)]
    return pd.DataFrame(rows)


def test_speaker_split_is_disjoint_and_sex_balanced(fake_corpus):
    df = speaker_split(fake_corpus)
    check_speaker_disjoint(df)
    assert set(df.split) == {"train", "val", "test"}
    for s in ["val", "test"]:
        actors = df[df.split == s].drop_duplicates("actor")
        assert 0.3 < (actors.sex == "Male").mean() < 0.7


def test_cv_tests_every_speaker_exactly_once(fake_corpus):
    tested = []
    for fold in range(5):
        df = cv_speaker_split(fake_corpus, fold, n_folds=5)
        check_speaker_disjoint(df)
        tested += df.loc[df.split == "test", "actor"].unique().tolist()
    assert sorted(tested) == sorted(fake_corpus.actor.unique())


def test_random_clip_split_leaks_speakers(fake_corpus):
    """The 'wrong' split used for the leakage experiment must be caught by the check."""
    with pytest.raises(AssertionError, match="appear in both"):
        check_speaker_disjoint(random_clip_split(fake_corpus))


@pytest.mark.skipif(not (CREMA_DIR / "AudioWAV").exists(), reason="CREMA-D not downloaded")
def test_saved_crema_split():
    from src.dataset import load_split
    df = load_split(build_index())
    assert len(df) == 7442 and df.actor.nunique() == 91
    assert df.groupby("split").actor.nunique().to_dict() == {"train": 65, "val": 13, "test": 13}
    check_speaker_disjoint(df)
    assert np.all(df.split.notna())
