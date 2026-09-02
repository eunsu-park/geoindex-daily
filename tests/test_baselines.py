import numpy as np
import pandas as pd
import pytest

from geoindex_daily.baselines import (climatology, persistence, recurrence, recurrence_lag,
                                      target)


def periodic(period=27, n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(3, 40, size=period)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.tile(base, n // period + 1)[:n], index=idx)


@pytest.mark.parametrize("h", [1, 7, 26, 27, 28, 45, 54, 60])
def test_recurrence_lag_never_negative_and_on_rotation(h):
    lag = recurrence_lag(h)
    assert lag >= 0
    assert (lag + h) % 27 == 0
    assert recurrence_lag(h, rotations_back=1) == lag + 27


@pytest.mark.parametrize("h", [1, 7, 27, 45, 60])
def test_recurrence_is_exact_on_periodic_series(h):
    s = periodic()
    m = pd.concat([target(s, h), recurrence(s, h)], axis=1).dropna()
    assert len(m) > 200
    np.testing.assert_allclose(m.iloc[:, 0], m.iloc[:, 1])


def test_recurrence_uses_no_future_values():
    s = periodic()
    s.iloc[300:] = 1000.0  # poison the future
    f = recurrence(s, 60)
    # forecasts issued before day 300 - lag must not see the poisoned values
    lag = recurrence_lag(60)
    assert (f.iloc[: 300 + lag].dropna() < 1000).all()
    assert (f.iloc[300 + lag:] == 1000).all()


def test_persistence_and_climatology_alignment():
    s = periodic()
    assert persistence(s, 5).equals(s.rename("pers_h5"))
    c = climatology(s, 5, train_end="2020-03-01")
    assert c.index.equals(s.index)
    assert np.isclose(c.iloc[0], s[: "2020-03-01"].mean())


def test_target_alignment():
    s = pd.Series(np.arange(10.0), index=pd.date_range("2020-01-01", periods=10, freq="D"))
    y = target(s, 3)
    assert y.iloc[0] == 3.0 and np.isnan(y.iloc[-1])
