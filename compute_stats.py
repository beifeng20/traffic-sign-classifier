"""Compute per-channel mean/std of the training split (used by tsc/data.py normalize)."""
from tsc.data import _load, split_by_track

x, _, paths = _load("train", "data")
tr, _ = split_by_track(paths)
pixels = x[tr].reshape(-1, 3).astype("float64") / 255.0
print("MEAN =", tuple(round(float(v), 4) for v in pixels.mean(0)))
print("STD  =", tuple(round(float(v), 4) for v in pixels.std(0)))
