import numpy as np
import pandas as pd
import torch

from geoindex_daily.ap3h import BINS, TS_COLS, cme_days, gaussian_nll, scores, windows_3h


def _f3(days=10):
    t = pd.date_range("2020-01-01", periods=BINS * days, freq="3h")
    return pd.DataFrame({"ap_t": np.arange(len(t), dtype=float), "xray_mean_log": 0.0, "xray_max_log": 1.0}, index=t)


def test_windows_end_at_the_issue_day_and_target_the_next_day():
    X, Y = windows_3h(_f3(), pd.DatetimeIndex(["2020-01-03"]), L=2, H=8)
    assert X.shape == (1, 16, len(TS_COLS)) and Y.shape == (1, 8)
    assert X[0, 0, 0] == BINS * 1 and X[0, -1, 0] == BINS * 3 - 1      # bins of 01-02 00 UT .. 01-03 21 UT
    assert Y[0, 0] == BINS * 3                                         # first bin of 01-04


def test_cme_window_is_the_issue_day_and_two_before(tmp_path):
    cat = pd.DataFrame({"cme_date": pd.to_datetime(["2020-01-05"]), "earth_directed": [True]})
    cat.to_parquet(tmp_path / "c.parquet")
    days = pd.date_range("2020-01-04", "2020-01-08")
    assert list(cme_days(days, tmp_path / "c.parquet")) == [False, True, True, True, False]


def test_nll_is_smaller_for_the_right_mean_and_scores_average_eight_bins():
    y = torch.zeros(4, 8)
    good = torch.cat([torch.zeros(4, 8), torch.zeros(4, 8)], 1)
    bad = torch.cat([torch.ones(4, 8), torch.zeros(4, 8)], 1)
    assert gaussian_nll(good, y) < gaussian_nll(bad, y)
    yy = np.full((2, 8), 3.0)
    s = scores(np.c_[yy[:, :4] * 0, yy[:, 4:] * 2], np.log1p(np.full((2, 8), 3.0)), np.full((2, 8), 0.1))
    assert np.isclose(s["mae_day"], 0.0) and np.isclose(s["mae_3h"], 3.0)
