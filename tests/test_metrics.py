import numpy as np
import pandas as pd

from geoindex_daily.metrics import corr, log_mae, mae, rmse, score, storm_scores


def _pair():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    y = pd.Series([5, 10, 40, 8, 35, np.nan], index=idx)
    f = pd.Series([6, 8, 12, 8, 50, 7], index=idx)
    return y, f


def test_basic_scores_ignore_nan_rows():
    y, f = _pair()
    assert np.isclose(mae(y, f), np.mean([1, 2, 28, 0, 15]))
    assert rmse(y, f) >= mae(y, f)
    assert log_mae(y, f) < mae(y, f)
    assert -1 <= corr(y, f) <= 1


def test_storm_contingency():
    y, f = _pair()
    s = storm_scores(y, f, threshold=30)
    # events: days 3 (40) and 5 (35); forecasts >= 30: day 5 only → 1 hit, 1 miss, 0 false
    assert s["n_events"] == 2 and s["n_forecast"] == 1
    assert np.isclose(s["pod"], 0.5) and np.isclose(s["far"], 0.0) and np.isclose(s["csi"], 0.5)


def test_score_is_flat_dict():
    y, f = _pair()
    d = score(y, f)
    assert d["n"] == 5
    assert {"mae", "rmse", "log_mae", "corr", "pod", "far", "csi"} <= d.keys()
