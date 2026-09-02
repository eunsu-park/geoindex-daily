"""Sliding (input, target) windows over a daily series, and date-range splits.

A sample issued on day t has input `s[t-input_days+1 .. t]` and target
`s[t+1 .. t+output_days]`. Samples whose window contains a NaN are dropped, so the
arrays are dense. Splits are by issue date and inclusive on both ends.
"""
import numpy as np
import pandas as pd


def make_windows(s: pd.Series, input_days: int = 30, output_days: int = 60,
                 stride: int = 1) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Returns `(X (N, input_days), Y (N, output_days), issue_dates (N,))`."""
    v = s.to_numpy(dtype=float)
    n = len(v)
    xs, ys, dates = [], [], []
    for t in range(input_days - 1, n - output_days, stride):
        x = v[t - input_days + 1: t + 1]
        y = v[t + 1: t + 1 + output_days]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        xs.append(x); ys.append(y); dates.append(s.index[t])
    if not xs:
        return (np.empty((0, input_days)), np.empty((0, output_days)),
                pd.DatetimeIndex([], name="issue_date"))
    return np.stack(xs), np.stack(ys), pd.DatetimeIndex(dates, name="issue_date")


def split_mask(issue_dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return np.asarray((issue_dates >= lo) & (issue_dates <= hi), dtype=bool)


def split(issue_dates: pd.DatetimeIndex, splits: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """`{name: boolean mask}` for a config `splits` block of `{name: [start, end]}`."""
    return {name: split_mask(issue_dates, lo, hi) for name, (lo, hi) in splits.items()}
