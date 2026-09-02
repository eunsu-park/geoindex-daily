import numpy as np
import pandas as pd

from geoindex_daily.windows import make_windows, split


def test_window_shapes_and_alignment():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2020-01-01", periods=100, freq="D"))
    X, Y, dates = make_windows(s, input_days=30, output_days=60)
    # issue days t = 29 .. 39 (target must end by day 99) → 11 samples
    assert X.shape == (11, 30) and Y.shape == (11, 60) and len(dates) == 11
    assert X[0, -1] == 29.0 and Y[0, 0] == 30.0 and dates[0] == pd.Timestamp("2020-01-30")
    assert Y[-1, -1] == 99.0


def test_windows_with_nan_are_dropped():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2020-01-01", periods=100, freq="D"))
    s.iloc[50] = np.nan
    X, Y, dates = make_windows(s, 30, 60)
    assert len(dates) == 0  # every 90-day window covers day 50
    X, Y, dates = make_windows(s, 10, 5)
    assert not np.isnan(X).any() and not np.isnan(Y).any()
    assert all(not (d - pd.Timedelta(days=9) <= s.index[50] <= d + pd.Timedelta(days=5)) for d in dates)


def test_split_masks_are_disjoint_and_inclusive():
    dates = pd.date_range("2019-12-30", periods=5, freq="D")
    m = split(dates, {"train": ["2019-01-01", "2019-12-31"], "val": ["2020-01-01", "2020-01-02"]})
    assert m["train"].tolist() == [True, True, False, False, False]
    assert m["val"].tolist() == [False, False, True, True, False]
