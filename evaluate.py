"""Evaluate the best checkpoint on the official GTSRB test set (12,630 images).

Writes: results/metrics.json, results/confusion_matrix.png, results/errors.png, results/training_curves.png
"""
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from tsc.data import MEAN, STD, load_splits
from tsc.labels import CLASS_NAMES
from tsc.model import SignNet


@torch.no_grad()
def predict(model, loader):
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x).argmax(1))
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main(run: str = "runs/signnet", out: str = "results") -> None:
    out = Path(out)
    out.mkdir(exist_ok=True)
    _, _, test_ds = load_splits()
    model = SignNet()
    model.load_state_dict(torch.load(Path(run) / "best.pt", map_location="cpu"))
    model.eval()

    t0 = time.time()
    pred, true = predict(model, DataLoader(test_ds, batch_size=512))
    ms_per_img = (time.time() - t0) / len(true) * 1000

    acc = float((pred == true).mean())
    macro_f1 = float(f1_score(true, pred, average="macro"))
    report = classification_report(true, pred, output_dict=True, zero_division=0)
    per_class = sorted(((report[str(c)]["recall"], c) for c in range(43)))
    metrics = {
        "test_images": int(len(true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "errors": int((pred != true).sum()),
        "cpu_ms_per_image_batch512": round(ms_per_img, 3),
        "worst_classes_by_recall": [
            {"class": c, "name": CLASS_NAMES[c], "recall": round(r, 4)} for r, c in per_class[:5]],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # confusion matrix, row-normalized, log color scale so rare mistakes are visible
    cm = confusion_matrix(true, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.clip(cm, 1e-3, 1), cmap="viridis", norm=matplotlib.colors.LogNorm(1e-3, 1))
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"GTSRB test confusion matrix (accuracy {acc:.2%})")
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class (log scale)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # grid of misclassified test images
    wrong = np.where(pred != true)[0][:24]
    fig, axes = plt.subplots(4, 6, figsize=(12, 8.5))
    mean, std = np.array(MEAN), np.array(STD)
    for ax, i in zip(axes.flat, wrong):
        img = test_ds[i][0].permute(1, 2, 0).numpy() * std + mean
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"true {true[i]} / pred {pred[i]}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Misclassified test images (class ids, see tsc/labels.py)")
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=110)
    plt.close(fig)

    # training curves from the log
    rows = list(csv.DictReader(open(Path(run) / "log.csv")))
    ep = [int(r["epoch"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    a1.plot(ep, [float(r["val_loss"]) for r in rows], label="validation")
    a1.set_title("loss")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(ep, [float(r["train_acc"]) for r in rows], label="train")
    a2.plot(ep, [float(r["val_acc"]) for r in rows], label="validation")
    a2.set_title("accuracy")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
