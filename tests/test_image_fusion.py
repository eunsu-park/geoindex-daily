import numpy as np
import pandas as pd
import torch

from geoindex_daily.frames import cache_path
from geoindex_daily.image_data import FrameWindowDataset, check_cache, frame_paths
from geoindex_daily.models.image_fusion import ImageBranch, ImageFusion


def test_frame_paths_order_and_count():
    ps = frame_paths("/c", pd.Timestamp("2024-05-10"), L=2, slot_hours=(0, 12), channels=("aia193", "lasco_c2"))
    assert len(ps) == 2 and len(ps[0]) == 4
    assert [p.stem for p in ps[0]] == ["20240509_00", "20240509_12", "20240510_00", "20240510_12"]
    assert "lasco_c2" in str(ps[1][0])


def test_dataset_reads_and_downsamples(tmp_path):
    days = pd.DatetimeIndex(["2024-05-10", "2024-05-11"])
    for d in pd.date_range("2024-05-09", "2024-05-11"):
        for h in (0, 12):
            for ch in ("aia193", "hmi_m45"):
                p = cache_path(tmp_path, ch, d + pd.Timedelta(hours=h))
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, np.full((64, 64), 100, np.uint8))
    assert check_cache(tmp_path, days, 2, (0, 12), ("aia193", "hmi_m45")).ok.all()
    assert check_cache(tmp_path, days, 3, (0, 12), ("aia193", "hmi_m45")).ok.tolist() == [False, True]
    ds = FrameWindowDataset(tmp_path, days, 2, (0, 12), ("aia193", "hmi_m45"),
                            ts=np.zeros((2, 2, 3)), y=np.array([1.0, 2.0]), size=32)
    img, ts, y = ds[1]
    assert img.shape == (2, 4, 32, 32) and img.dtype == torch.uint8 and int(img[0, 0, 0, 0]) == 100
    assert ts.shape == (2, 3) and float(y) == 2.0


def test_fusion_forward_shapes():
    m = ImageFusion(n_channels=4, ts_features=3, widths=(8, 8, 16, 16), ts_width=8, hidden=16)
    img = torch.randint(0, 256, (2, 4, 4, 128, 128), dtype=torch.uint8)
    out = m(img, torch.zeros(2, 2, 3))
    assert out.shape == (2,)
    only = ImageFusion(n_channels=4, ts_features=0, widths=(8, 8, 16, 16))
    assert only(img).shape == (2,)
    b = ImageBranch(4, (8, 8, 16, 16))
    assert b(img).shape == (2, 16)
    m.image.set_normalisation([0.5] * 4, [0.2] * 4)
    assert torch.isfinite(m(img, torch.zeros(2, 2, 3))).all()
