"""Collect reports/results/*.json into one Markdown table (reports/results/summary.md).
Runs that differ only by seed are grouped and reported as mean ± std."""
import json
import re
from collections import defaultdict

import numpy as np

from src.dataset import ROOT

groups = defaultdict(list)
for f in sorted((ROOT / "reports" / "results").glob("*.json")):
    r = json.loads(f.read_text())
    if not isinstance(r, dict) or "history" not in r:  # ensembles.json, final.json: not training runs
        continue
    groups[re.sub(r"(_f\d+)?_s\d+$", "", r["name"])].append(r)  # seeds and CV folds


def fmt(values):
    values = 100 * np.array(values)
    return f"{values.mean():.1f} ± {values.std():.1f}" if len(values) > 1 else f"{values[0]:.1f}"


rows = []
for name, runs in groups.items():
    rows.append(f"| {name} | {len(runs)} | {runs[0]['params']:,} | "
                f"{fmt([r['val_uar'] for r in runs])} | {fmt([r['test']['accuracy'] for r in runs])} | "
                f"{fmt([r['test']['uar'] for r in runs])} | {fmt([r['test']['macro_f1'] for r in runs])} | "
                f"{np.mean([r['train_seconds'] for r in runs]) / 60:.1f} |")

table = "\n".join([
    "| run | runs (seeds/folds) | trainable params | val UAR % | test acc % | test UAR % | test macro-F1 % | train min |",
    "|---|---|---|---|---|---|---|---|",
    *rows,
])
(ROOT / "reports" / "results" / "summary.md").write_text(table + "\n")
print(table)
