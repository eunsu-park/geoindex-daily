import numpy as np
import pytest

from geoindex_daily.encoders.moment import pack


def test_pack_right_aligns_window_and_mask():
    X = np.arange(6.0).reshape(2, 3)
    x, mask = pack(X, seq_len=8)
    assert x.shape == (2, 1, 8) and mask.shape == (2, 8)
    assert x[0, 0, 5:].tolist() == [0.0, 1.0, 2.0] and x[0, 0, :5].abs().sum() == 0
    assert mask[0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]


def test_pack_rejects_window_longer_than_context():
    with pytest.raises(ValueError):
        pack(np.zeros((1, 9)), seq_len=8)
