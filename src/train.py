"""Train one model, keep the best epoch on validation speakers, evaluate on test speakers.

    python -m src.train --arch scratch
    python -m src.train --arch resnet18
    python -m src.train --arch scratch --split random   # leakage experiment
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.audio import fix_length
from src.dataset import EMOTION_NAMES, ROOT, SERDataset, load_split, load_waveform_cache, random_clip_split
from src.models import SERModel, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["scratch", "resnet18", "cnn14"], required=True)
    p.add_argument("--split", choices=["speaker", "random"], default="speaker")
    p.add_argument("--no-pretrained", action="store_true", help="resnet18 with random init (ablation)")
    p.add_argument("--freeze", action="store_true", help="pretrained models: train the head only")
    p.add_argument("--upsample", type=int, default=1, help="resnet18: enlarge the spectrogram by this factor")
    p.add_argument("--width", type=int, default=32, help="scratch: channels of the first block")
    p.add_argument("--mixup", type=float, default=0.0, help="mixup Beta(a, a) parameter, 0 = off")
    p.add_argument("--noise", action="store_true", help="add white noise at a random 10-40 dB SNR")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=0,
                   help="early stopping patience in epochs, 0 = off (OneCycle needs the full schedule)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    return p.parse_args()


def run_name(args):
    if args.name:
        return args.name
    name = args.arch
    if args.arch != "scratch":
        name += "_nopretrain" if args.no_pretrained else "_frozen" if args.freeze else ""
    if args.upsample > 1:
        name += f"_up{args.upsample}"
    if args.arch == "scratch" and args.width != 32:
        name += f"_w{args.width}"
    if args.mixup:
        name += f"_mixup{args.mixup:g}"
    if args.noise:
        name += "_noise"
    if args.epochs != 60:
        name += f"_ep{args.epochs}"
    return f"{name}_{args.split}split_s{args.seed}"


def augment_waveform(x, y, args, n_classes):
    """Waveform-level augmentation on the GPU. Returns the batch and soft targets."""
    target = nn.functional.one_hot(y, n_classes).float()
    if args.noise:
        snr_db = torch.empty(len(x), 1, device=x.device).uniform_(10, 40)
        power = x.pow(2).mean(dim=1, keepdim=True)
        noisy = x + torch.randn_like(x) * (power / 10 ** (snr_db / 10)).sqrt()
        x = torch.where(torch.rand(len(x), 1, device=x.device) < 0.5, noisy, x)
    if args.mixup:
        lam = float(np.random.beta(args.mixup, args.mixup))
        perm = torch.randperm(len(x), device=x.device)
        x = lam * x + (1 - lam) * x[perm]
        target = lam * target + (1 - lam) * target[perm]
    return x, target


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "uar": balanced_accuracy_score(y_true, y_pred),  # unweighted average recall
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def save_figures(history, cm, name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h["epoch"] for h in history]
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train accuracy")
    axes[1].plot(ep, [h["val_uar"] for h in history], label="val UAR (unseen speakers)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1); axes[1].legend()
    fig.suptitle(name); fig.tight_layout()
    fig.savefig(out_dir / f"{name}_curves.png", dpi=110); plt.close(fig)

    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(cm)), EMOTION_NAMES, rotation=45); ax.set_yticks(range(len(cm)), EMOTION_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name}\ntest set, row-normalised")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion.png", dpi=110); plt.close(fig)


def main():
    args = parse_args()
    name = run_name(args)
    lr = args.lr or {"scratch": 1e-3, "resnet18": 1e-3 if args.freeze else 3e-4, "cnn14": 1e-3}[args.arch]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_split()
    if args.split == "random":
        df = random_clip_split(df.drop(columns="split"), seed=args.seed)
    waves = load_waveform_cache(df)
    idx = {s: df.index[df.split == s].to_numpy() for s in ["train", "val", "test"]}

    def loader(split, train):
        ds = SERDataset([waves[i] for i in idx[split]], df.label[idx[split]], train=train, seed=args.seed)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, drop_last=train)

    train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)

    kwargs = {"scratch": {"width": args.width},
              "resnet18": {"pretrained": not args.no_pretrained, "freeze_backbone": args.freeze,
                           "upsample": args.upsample},
              "cnn14": {"pretrained": not args.no_pretrained, "freeze_backbone": args.freeze}}[args.arch]
    model = SERModel(args.arch, **kwargs).to(device)
    # normalisation statistics from the training clips only
    model.frontend.fit(torch.stack([torch.from_numpy(fix_length(waves[i])) for i in idx["train"]]))

    opt = torch.optim.AdamW(model.param_groups(lr), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=args.epochs * len(train_dl), pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"{name}: {count_params(model):,} trainable params, lr {lr}, "
          f"train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])} clips on {device}")
    ckpt_dir = ROOT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    best_uar, best_epoch, history = -1.0, 0, []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze:  # keep the pretrained BatchNorm statistics
            (model.net.backbone if args.arch == "resnet18" else model.net).eval()
        tot_loss, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            x, target = augment_waveform(x, y, args, len(EMOTION_NAMES))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot_loss += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); n += len(y)

        val_pred, val_true = predict(model, val_dl, device)
        val = metrics(val_true, val_pred)
        history.append({"epoch": epoch, "train_loss": tot_loss / n, "train_acc": correct / n,
                        "val_acc": val["accuracy"], "val_uar": val["uar"]})
        improved = val["uar"] > best_uar
        if improved:
            best_uar, best_epoch = val["uar"], epoch
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "model_kwargs": kwargs,
                        "classes": EMOTION_NAMES, "epoch": epoch, "val_uar": best_uar}, ckpt_path)
        print(f"ep {epoch:3d}  loss {tot_loss / n:.3f}  train acc {correct / n:.3f}  "
              f"val acc {val['accuracy']:.3f}  val UAR {val['uar']:.3f}{'  *' if improved else ''}"
              f"  ({time.time() - t0:.0f}s)")
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    y_pred, y_true = predict(model, test_dl, device)
    test = metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class = f1_score(y_true, y_pred, average=None)
    print(f"\nbest epoch {best_epoch} (val UAR {best_uar:.3f})")
    print(f"TEST  acc {test['accuracy']:.3f}  UAR {test['uar']:.3f}  macro-F1 {test['macro_f1']:.3f}")
    print("per-class F1: " + "  ".join(f"{c} {f:.2f}" for c, f in zip(EMOTION_NAMES, per_class)))

    out_dir = ROOT / "reports" / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "reports" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    result = {"name": name, "args": vars(args), "lr": lr, "params": count_params(model),
              "best_epoch": best_epoch, "val_uar": best_uar, "test": test,
              "test_per_class_f1": dict(zip(EMOTION_NAMES, per_class.tolist())),
              "confusion_matrix": cm.tolist(), "train_seconds": time.time() - t0,
              "test_actors": sorted(df.actor[idx["test"]].unique().tolist()) if args.split == "speaker" else None,
              "history": history}
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
    save_figures(history, cm, name, fig_dir)
    print(f"saved {ckpt_path.relative_to(ROOT)}, reports/results/{name}.json and figures")


if __name__ == "__main__":
    main()
