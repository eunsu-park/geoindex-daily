import numpy as np
import torch

from geoindex_daily.encoders.surya import SURYA_CHANNELS, pool_tokens, scaler_arrays, transform


def test_transform_matches_signum_log_standardize():
    scalers = {ch: {"mean": 1.0, "std": 2.0, "epsilon": 0.5, "sl_scale_factor": 10.0} for ch in SURYA_CHANNELS}
    sc = scaler_arrays(scalers)
    raw = np.zeros((13, 2, 2), dtype=np.float32)
    raw[0, 0, 0] = 1.0     # → sign(10)·log1p(10) = 2.3979 → (2.3979-1)/2.5
    raw[1, 0, 0] = -1.0    # → -2.3979 → (-2.3979-1)/2.5
    out = transform(raw, sc)
    assert out.dtype == np.float32
    assert np.isclose(out[0, 0, 0], (np.log1p(10) - 1) / 2.5)
    assert np.isclose(out[1, 0, 0], (-np.log1p(10) - 1) / 2.5)
    assert np.isclose(out[2, 0, 0], -1 / 2.5)  # raw 0 → (0-1)/2.5, not 0


def test_pool_tokens_shapes_and_time_average():
    side, D, T = 8, 4, 2
    tok = torch.arange(T * side * side * D, dtype=torch.float32).view(1, T * side * side, D)
    out = pool_tokens(tok, grid=2, patch=16, img=side * 16)
    assert out["mean"].shape == (D,) and out["first"].shape == (D,) and out["grid"].shape == (2, 2, D)
    assert np.allclose(out["first"], tok[0, 0].numpy())
    # grid averages over the two time steps then over 4x4 spatial blocks
    spatial = tok.view(T, side, side, D).mean(0)
    assert np.allclose(out["grid"][0, 0], spatial[:4, :4].mean((0, 1)).numpy())


def test_pool_tokens_rejects_bad_length():
    try:
        pool_tokens(torch.zeros(1, 100, 4), grid=2, patch=16, img=128)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
