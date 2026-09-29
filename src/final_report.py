"""Final numbers and figures for the README (reports/results/final.json, reports/figures/final_*.png).

    python -m src.final_report

- the deployed ensemble (the first entry of app/models/registry.json) on the 13 test speakers: UAR,
  per-class F1, confusion matrix, and how confident it is for each emotion
- the 5-fold leave-speakers-out CV checkpoints: pooled UAR over all 7,442 clips and per-actor UAR,
  so that every one of the 91 actors is scored as an unseen speaker
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.dataset import EMOTION_NAMES, ROOT, cv_speaker_split, load_split, load_waveform_cache
from src.evaluate import predict_proba
from src.models import Ensemble, load_checkpoint

FIG = ROOT / "reports" / "figures"
BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CV_MODELS = {"scratch": "scratch_cv5split_f{}_s0", "dilated": "dilated_w48_cv5split_f{}_s0"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def per_actor_uar(actors, y, pred):
    return {int(a): balanced_accuracy_score(y[actors == a], pred[actors == a]) for a in np.unique(actors)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_split()
    waves = load_waveform_cache(df)
    out = {}

    # --- deployed ensemble on the fixed test speakers (sliding windows, as in the app)
    deployed = json.loads((ROOT / "app" / "models" / "registry.json").read_text())[0]
    members = [load_checkpoint(ROOT / "app" / "models" / f"{c}.pt", device) for c in deployed["checkpoints"]]
    model = Ensemble(members).eval() if len(members) > 1 else members[0]
    test = df.index[df.split == "test"]
    y = df.label[test].to_numpy()
    probs = predict_proba(model, [waves[i] for i in test], device, windows=True)
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred)
    out["ensemble"] = {"members": [m.name for m in members], "test_uar": balanced_accuracy_score(y, pred),
                       "test_accuracy": float((y == pred).mean()),
                       "per_class_f1": dict(zip(EMOTION_NAMES, f1_score(y, pred, average=None).round(3).tolist())),
                       "per_class_recall": dict(zip(EMOTION_NAMES, (cm.diagonal() / cm.sum(1)).round(3).tolist())),
                       "confusion_matrix": cm.tolist(),
                       # for the clips of each true emotion: mean probability given to the right answer,
                       # and share of clips where the model's top probability is below 50 %
                       "per_class_prob_of_true": dict(zip(EMOTION_NAMES, [round(float(probs[y == k, k].mean()), 3)
                                                                          for k in range(6)])),
                       "per_class_share_top_below_50": dict(zip(EMOTION_NAMES, [
                           round(float((probs[y == k].max(1) < 0.5).mean()), 3) for k in range(6)]))}

    # --- leave-speakers-out CV: every actor is a test speaker once
    base = df.drop(columns="split")
    for arch, pattern in CV_MODELS.items():
        ys, preds, actors = [], [], []
        for fold in range(5):
            d = cv_speaker_split(base, fold)
            idx = d.index[d.split == "test"]
            m = load_checkpoint(ROOT / "checkpoints" / f"{pattern.format(fold)}.pt", device)
            preds.append(predict_proba(m, [waves[i] for i in idx], device, windows=False).argmax(1))
            ys.append(d.label[idx].to_numpy()); actors.append(d.actor[idx].to_numpy())
        ys, preds, actors = map(np.concatenate, (ys, preds, actors))
        spk = per_actor_uar(actors, ys, preds)
        out[f"cv_{arch}"] = {"pooled_uar": balanced_accuracy_score(ys, preds), "n_clips": len(ys),
                             "per_actor_uar": spk,
                             "per_actor_quantiles": dict(zip(["min", "q25", "median", "q75", "max"],
                                                             np.quantile(list(spk.values()), [0, .25, .5, .75, 1])
                                                             .round(3).tolist())),
                             "confusion_matrix": confusion_matrix(ys, preds).tolist()}

    (ROOT / "reports" / "results" / "final.json").write_text(json.dumps(out, indent=1))
    plot_confusion(np.array(out["ensemble"]["confusion_matrix"]))
    plot_speakers({a: out[f"cv_{a}"]["per_actor_uar"] for a in CV_MODELS}, df)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_actor_uar", "confusion_matrix")}
                      for k, v in out.items()}, indent=1))


def plot_confusion(cm):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 5), facecolor=SURFACE)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(6), EMOTION_NAMES, rotation=35, ha="right"); ax.set_yticks(range(6), EMOTION_NAMES)
    ax.set_xlabel("predicted", color=MUTED); ax.set_ylabel("true", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Deployed ensemble, 13 test speakers\n% of each true emotion (rows sum to 100)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "final_confusion.png", dpi=130); plt.close(fig)


def plot_speakers(per_actor, df):
    """One dot per actor, per model: how much the score depends on who is speaking."""
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURFACE)
    style(ax)
    rng = np.random.default_rng(0)
    names = {"scratch": "CNN from scratch", "dilated": "dilated CNN"}
    for row, (arch, spk) in enumerate(per_actor.items()):
        vals = 100 * np.array(list(spk.values()))
        ax.scatter(vals, row + rng.uniform(-0.18, 0.18, len(vals)), s=22, color=BLUE, alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)
        med = np.median(vals)
        ax.plot([med, med], [row - 0.3, row + 0.3], color=INK, lw=2, zorder=4)
        ax.text(med, row + 0.36, f"median {med:.0f} %", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(range(len(per_actor)), [names[a] for a in per_actor], color=INK, fontsize=10)
    ax.set_ylim(-0.6, len(per_actor) - 0.25)
    ax.set_xlim(0, 100); ax.set_xlabel("UAR for one actor (%), actor unseen in training", color=MUTED)
    ax.axvline(100 / 6, color=MUTED, lw=1, ls=":"); ax.text(100 / 6 + 1, -0.55, "chance", fontsize=8, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_title("Leave-speakers-out cross-validation: each dot is one of the 91 actors", fontsize=10,
                 color=INK, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "final_cv_speakers.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
