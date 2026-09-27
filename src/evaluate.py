"""Re-evaluate saved checkpoints: single models and probability-averaging ensembles,
with a single centred 3 s window (as in training) or averaged sliding windows (as in the app).

    python -m src.evaluate scratch_speakersplit_s0 cnn14_ep40_speakersplit_s0 ...

Choose ensembles on the val column; the test column is only for reporting.
"""
import argparse
import itertools
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from src.audio import fix_length, sliding_windows
from src.dataset import EMOTION_NAMES, ROOT, load_split, load_waveform_cache
from src.models import SERModel


def load_model(name, device):
    ckpt = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location=device)
    kwargs = {**ckpt["model_kwargs"], **({"pretrained": False} if "pretrained" in ckpt["model_kwargs"] else {})}
    model = SERModel(ckpt["arch"], len(EMOTION_NAMES), **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def predict_proba(model, waves, device, windows: bool, batch_size: int = 128):
    if not windows:
        out = []
        for i in range(0, len(waves), batch_size):
            x = torch.stack([torch.from_numpy(fix_length(w)) for w in waves[i:i + batch_size]]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out.append(torch.softmax(model(x).float(), 1).cpu())
        return torch.cat(out).numpy()
    wins = [sliding_windows(w) for w in waves]
    owner = np.concatenate([[i] * len(w) for i, w in enumerate(wins)])
    flat = torch.cat(wins)
    probs = []
    for i in range(0, len(flat), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            probs.append(torch.softmax(model(flat[i:i + batch_size].to(device)).float(), 1).cpu())
    probs = torch.cat(probs).numpy()
    return np.stack([probs[owner == i].mean(0) for i in range(len(waves))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--max-ensemble", type=int, default=3)
    p.add_argument("--out", default=None, help="save the table as JSON under reports/results/")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_split()
    waves = load_waveform_cache(df)
    split = {s: df.index[df.split == s].to_numpy() for s in ["val", "test"]}
    y = {s: df.label[split[s]].to_numpy() for s in split}

    probs = {}  # (model, split, windows) -> (N, 6)
    for name in args.models:
        model = load_model(name, device)
        for s in split:
            for w in (False, True):
                probs[name, s, w] = predict_proba(model, [waves[i] for i in split[s]], device, w)
        del model

    rows = []
    for k in range(1, min(args.max_ensemble, len(args.models)) + 1):
        for combo in itertools.combinations(args.models, k):
            for w in (False, True):
                r = {"models": list(combo), "windows": w}
                for s in split:
                    avg = np.mean([probs[m, s, w] for m in combo], axis=0)
                    r[f"{s}_uar"] = balanced_accuracy_score(y[s], avg.argmax(1))
                rows.append(r)
    rows.sort(key=lambda r: -r["val_uar"])
    print(f"{'val UAR':>8} {'test UAR':>9}  windows  models")
    for r in rows[:25]:
        print(f"{100 * r['val_uar']:8.1f} {100 * r['test_uar']:9.1f}  {'yes' if r['windows'] else 'no ':7}  "
              + " + ".join(r["models"]))
    if args.out:
        (ROOT / "reports" / "results" / args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
