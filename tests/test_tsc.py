import numpy as np
import torch

from tsc.data import SignDataset, split_by_track, track_id
from tsc.labels import CLASS_NAMES
from tsc.model import SignNet


def test_model_output_shape():
    out = SignNet()(torch.zeros(4, 3, 32, 32))
    assert out.shape == (4, 43)


def test_track_id_parsing():
    assert track_id("data/gtsrb/GTSRB/Training/00014/00003_00029.ppm") == "00014/00003"


def test_split_has_no_shared_tracks():
    paths = [f"Training/{c:05d}/{t:05d}_{f:05d}.ppm" for c in range(3) for t in range(10) for f in range(30)]
    tr, va = split_by_track(paths, val_fraction=0.2)
    assert len(tr) + len(va) == len(paths)
    tracks_tr = {track_id(paths[i]) for i in tr}
    tracks_va = {track_id(paths[i]) for i in va}
    assert tracks_tr.isdisjoint(tracks_va)
    assert len(va) == 6 * 30  # 20% of 30 tracks, 30 frames each


def test_dataset_item_is_normalized_float():
    x = np.full((2, 32, 32, 3), 128, np.uint8)
    img, label = SignDataset(x, np.array([1, 2]), train=False)[0]
    assert img.dtype == torch.float32 and img.shape == (3, 32, 32)
    assert label.item() == 1


def test_43_class_names():
    assert len(CLASS_NAMES) == 43
