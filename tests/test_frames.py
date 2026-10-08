import numpy as np
import pandas as pd

from geoindex_daily.frames import (
    block_mean, cache_path, calibrate, lasco_running_difference, preprocess_aia, preprocess_hmi,
    preprocess_lasco, resize_to, to_uint8,
)


def test_block_mean_and_resize():
    a = np.arange(16, dtype=float).reshape(4, 4)
    b = block_mean(a, 2)
    assert b.shape == (2, 2) and b[0, 0] == np.mean([0, 1, 4, 5])
    assert resize_to(np.zeros((4096, 4096), np.float32), 1024).shape == (1024, 1024)
    assert resize_to(np.ones((512, 512)), 1024).shape == (1024, 1024)
    assert resize_to(a, 4) is a


def test_to_uint8_clips_and_scales():
    assert to_uint8(np.array([-1.0, 0.0, 5.0, 10.0, 20.0]), 0, 10).tolist() == [0, 0, 127, 255, 255]


def test_aia_log_rate_and_nan_safe():
    d = np.full((4096, 4096), 2000.0, np.float32)
    d[0, 0] = np.nan
    out = preprocess_aia(d, exptime=2.0, hi=np.log1p(1000.0))
    assert out.shape == (1024, 1024) and out.dtype == np.uint8
    assert out[5, 5] == 255 and out[0, 0] < 255        # 1000 DN/s hits hi; the NaN block is lower


def test_hmi_flip_and_scale():
    d = np.zeros((4096, 4096), np.float64)
    d[:, 2048:] = 1500.0          # positive field on the right half
    d[0, 0] = np.nan
    up = preprocess_hmi(d, crota2=179.9)
    assert up[512, 100] == 255 and up[512, 900] == 127   # flipped: positive now on the left; zero → 127/128
    noflip = preprocess_hmi(d, crota2=0.1)
    assert noflip[512, 900] == 255 and noflip[512, 100] == 127


def test_lasco_and_running_difference():
    cur = np.full((1024, 1024), 2500, np.int16)
    prev = np.full((1024, 1024), 2500, np.int16)
    cur[100, 100] = 5000                                 # a bright pixel appears
    out = preprocess_lasco(cur, 25.0, hi=np.log1p(100.0))
    assert out.dtype == np.uint8 and out[0, 0] >= 254   # 100 counts/s = hi → top of the range
    rd = lasco_running_difference(cur, 25.0, prev, 25.0, scale=60.0)
    assert rd[0, 0] in (127, 128) and rd[100, 100] == 255   # no change → mid grey; +100 counts/s saturates


def test_calibrate_uses_percentiles():
    f = {"aia193": [(np.full((8, 8), 200.0), {"EXPTIME": 2.0})], "lasco_c2": [(np.full((8, 8), 2500), {"EXPTIME": 25.0})]}
    c = calibrate(f, [(np.full((8, 8), 2600), 25.0, np.full((8, 8), 2500), 25.0)])
    assert np.isclose(c["aia193"]["hi"], np.log1p(100.0)) and np.isclose(c["lasco_c2"]["hi"], np.log1p(100.0))
    assert np.isclose(c["lasco_c2_rd"]["scale"], 4.0)


def test_cache_path_layout():
    p = cache_path("/tmp/c", "aia193", pd.Timestamp("2024-05-10 06:00"))
    assert str(p).endswith("c/aia193/2024/20240510_06.npy")
