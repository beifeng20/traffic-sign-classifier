"""GTSRB loading: images cached as 32x32 uint8 arrays, train/val split by track."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import GTSRB
from torchvision.transforms import v2

IMG_SIZE = 32
NUM_CLASSES = 43


def _load(split: str, root: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return images (N, 32, 32, 3) uint8, labels (N,) and file paths; cached in an .npz file."""
    cache = Path(root) / f"cache_{split}_{IMG_SIZE}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"], list(d["paths"])
    ds = GTSRB(root, split=split, download=True)
    xs, ys, paths = [], [], []
    for path, label in ds._samples:
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        xs.append(np.asarray(img, dtype=np.uint8))
        ys.append(label)
        paths.append(str(path))
    x, y = np.stack(xs), np.array(ys, dtype=np.int64)
    np.savez_compressed(cache, x=x, y=y, paths=np.array(paths))
    return x, y, paths


def track_id(path: str) -> str:
    """Training files are named <track>_<frame>.ppm inside a per-class folder.

    One track = ~30 frames of the SAME physical sign, so frames of one track must
    not be split between train and validation (that would leak near-duplicates).
    """
    p = Path(path)
    return f"{p.parent.name}/{p.name.split('_')[0]}"


def split_by_track(paths: list[str], val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.array([track_id(p) for p in paths])
    unique = np.unique(tracks)
    rng = np.random.default_rng(seed)
    val_tracks = set(rng.choice(unique, size=int(len(unique) * val_fraction), replace=False))
    is_val = np.array([t in val_tracks for t in tracks])
    return np.where(~is_val)[0], np.where(is_val)[0]


# channel mean/std of the GTSRB training set (computed once with compute_stats.py)
MEAN = (0.3382, 0.3099, 0.3198)
STD = (0.2727, 0.2598, 0.2660)

normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

# No horizontal flip: many signs are not symmetric ("turn left" would become "turn right").
augment = v2.Compose([
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
])


class SignDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool):
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # N, C, H, W uint8
        self.y = torch.from_numpy(y)
        self.train = train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        img = self.x[i]
        if self.train:
            img = augment(img)
        return normalize(img), self.y[i]


def load_splits(root: str = "data", val_fraction: float = 0.1, seed: int = 0):
    x, y, paths = _load("train", root)
    tr, va = split_by_track(paths, val_fraction, seed)
    xt, yt, _ = _load("test", root)
    return (SignDataset(x[tr], y[tr], train=True), SignDataset(x[va], y[va], train=False),
            SignDataset(xt, yt, train=False))
