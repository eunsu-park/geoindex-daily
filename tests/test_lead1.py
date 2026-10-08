import numpy as np
import pandas as pd

from geoindex_daily.lead1 import (
    block_bootstrap_diff, common_issue_days, feature_windows, ffill_features, split_masks,
)
from geoindex_daily.models.ts_cnn import TSCNN, predict, train_ts_cnn


def _df(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame({"ap": np.arange(n, dtype=float), "x": np.arange(n, dtype=float) * 10}, index=idx)


def test_feature_windows_are_past_only_and_target_is_next_day():
    df = _df()
    days = pd.DatetimeIndex(df.index[[5, 10]])
    X, y = feature_windows(df, ["ap", "x"], L=3, issue_days=days)
    assert X.shape == (2, 3, 2)
    assert X[0, :, 0].tolist() == [3, 4, 5] and X[0, :, 1].tolist() == [30, 40, 50]
    assert y.tolist() == [6, 11]


def test_common_issue_days_needs_history_target_and_images():
    df = _df(40)
    df.loc[df.index[10], "x"] = np.nan                      # a gap on day 10
    days = common_issue_days(df, ["ap", "x"], history_days=5, lead=1, period=("2020-01-01", "2020-12-31"))
    # days 0..3 lack history; days 10..14 contain the gap; the last day has no target
    assert df.index[4] in days and df.index[9] in days
    assert all(df.index[i] not in days for i in range(10, 15))
    assert df.index[15] in days and df.index[-1] not in days
    imgs = pd.DatetimeIndex(df.index[20:25])
    assert common_issue_days(df, ["ap", "x"], 5, 1, ("2020-01-01", "2020-12-31"), imgs).tolist() == imgs.tolist()


def test_ffill_is_limited():
    df = _df(10)
    df.loc[df.index[2:7], "x"] = np.nan
    out = ffill_features(df, ["x"], limit=3)
    assert out.x.iloc[2:5].notna().all() and out.x.iloc[5:7].isna().all()


def test_split_masks_by_range_and_by_year():
    days = pd.date_range("2011-01-01", "2014-12-31", freq="D")
    m = split_masks(days, {"train": ["2011-01-01", "2012-12-31"], "val": ["2013-01-01", "2013-12-31"],
                           "test": ["2014-01-01", "2014-12-31"]})
    assert m["train"].sum() == 731 and m["val"].sum() == 365 and m["test"].sum() == 365
    m = split_masks(days, {"test_years": [2012], "val_years": [2014]})
    assert m["test"].sum() == 366 and m["val"].sum() == 365 and m["train"].sum() == 730
    assert not (m["train"] & m["test"]).any()


def test_bootstrap_diff_sign_and_interval():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    y = rng.normal(10, 3, 400)
    good, bad = y + rng.normal(0, 1, 400), y + rng.normal(0, 3, 400)
    r = block_bootstrap_diff(y, good, bad, dates, n=200)
    assert r["d_mae"] < 0 and r["d_mae_hi"] < 0        # good has lower MAE, interval excludes zero
    assert r["d_corr"] > 0 and r["d_corr_lo"] > 0


def test_ts_cnn_fits_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 3, 2)).astype(np.float32)
    y = X[:, -1, 0] * 0.8 + 0.1 * X[:, 0, 1]
    model, hist = train_ts_cnn(X[:400], y[:400], X[400:], y[400:], seed=0, max_epochs=60,
                               patience=10, device="cpu")
    assert isinstance(model, TSCNN)
    p = predict(model, X[400:], device="cpu")
    assert np.corrcoef(p, y[400:])[0, 1] > 0.9
    assert hist["best_epoch"] <= hist["epochs"]
