"""Train SignNet on GTSRB (CPU is enough: ~1-2 min per epoch).

Example:
    python train.py --epochs 12
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tsc.data import load_splits
from tsc.model import SignNet, count_params


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss = 0, 0, 0.0
    for x, y in loader:
        out = model(x)
        loss += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss / total, correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="runs/signnet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_splits()
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)
    print(f"train {len(train_ds)}  val {len(val_ds)} images")

    model = SignNet()
    print(f"SignNet: {count_params(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # one-cycle: LR warms up, then anneals to ~0; usually converges faster than a constant LR
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    with open(out / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, seen, correct, run_loss = time.time(), 0, 0, 0.0
            for x, y in train_dl:
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                sched.step()
                run_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                seen += len(y)
            val_loss, val_acc = evaluate(model, val_dl)
            secs = time.time() - t0
            log.writerow([epoch, round(run_loss / seen, 4), round(correct / seen, 4),
                          round(val_loss, 4), round(val_acc, 4), round(secs, 1)])
            f.flush()
            print(f"epoch {epoch:2d}  train_loss {run_loss / seen:.3f}  train_acc {correct / seen:.3f}  "
                  f"val_loss {val_loss:.3f}  val_acc {val_acc:.4f}  ({secs:.0f}s)")
            if val_acc > best:
                best = val_acc
                torch.save(model.state_dict(), out / "best.pt")
    print(f"best val_acc {best:.4f}, weights saved to {out / 'best.pt'}")


if __name__ == "__main__":
    main()
