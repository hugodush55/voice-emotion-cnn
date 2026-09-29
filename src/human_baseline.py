"""How often do CREMA-D's own raters recognise the intended emotion from the
voice alone, and from the face alone? Uses the corpus' rating summary
(processedResults/summaryTable.csv: VoiceVote / FaceVote = majority answer of the
audio-only / video-only raters for each clip).

    python -m src.human_baseline
"""
import pandas as pd

from src.dataset import CREMA_DIR, EMOTION_NAMES, SPLIT_PATH

URL = ("https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/"
       "processedResults/summaryTable.csv")


def main():
    path = CREMA_DIR / "summaryTable.csv"
    votes = pd.read_csv(path if path.exists() else URL)
    votes["emotion"] = votes.FileName.str.split("_").str[2].str[0]  # A D F H N S
    votes["actor"] = votes.FileName.str[:4].astype(int)
    votes["strict"] = votes.VoiceVote == votes.emotion  # ties such as "A:N" count as wrong
    votes["with_ties"] = [e in v.split(":") for e, v in zip(votes.emotion, votes.VoiceVote)]

    split = pd.read_csv(SPLIT_PATH)
    test_actors = split.actor[split.split == "test"]
    for name, df in [("all 91 actors", votes), ("13 test actors", votes[votes.actor.isin(test_actors)])]:
        recall = df.groupby("emotion")[["strict", "with_ties"]].mean()
        recall.index = EMOTION_NAMES
        print(f"\n{name}: majority-vote UAR {100 * recall.strict.mean():.1f} % "
              f"({100 * recall.with_ties.mean():.1f} % if ties containing the right answer count)")
        print((100 * recall).round(1).T.to_string())

    # voice vs face: which emotions are carried by the voice, which by the face?
    modality = pd.DataFrame({m: (votes[f"{m}Vote"] == votes.emotion).groupby(votes.emotion).mean() * 100
                             for m in ["Voice", "Face", "MultiModal"]}).round(1)
    modality.index = EMOTION_NAMES
    print("\nmajority-vote recall % by modality (all actors):")
    print(modality.T.to_string())


if __name__ == "__main__":
    main()
