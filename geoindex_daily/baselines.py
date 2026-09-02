"""Reference forecasts for a daily index: climatology, persistence, solar-rotation recurrence.

Every function takes a daily series `s` (contiguous DatetimeIndex) and a lead `h` in
days, and returns a series aligned on the **issue date** t whose value is the forecast
for t + h. Only values at or before t are used, so the forecasts are honest for every h.

Recurrence is the one that is easy to get wrong: Ap(t + h − 27) is a *future* value
whenever h > 27. The forecast therefore reaches back to the latest rotation multiple
that is at or before the issue date, `k = ceil(h / 27)`, and uses Ap(t + h − 27 k).
"""
import math

import pandas as pd

ROTATION_DAYS = 27


def climatology(s: pd.Series, h: int, train_end: str | pd.Timestamp | None = None) -> pd.Series:
    """Constant forecast: the mean over the training period (all of `s` if unspecified)."""
    ref = s[: pd.Timestamp(train_end)] if train_end is not None else s
    return pd.Series(ref.mean(), index=s.index, name=f"clim_h{h}")


def persistence(s: pd.Series, h: int) -> pd.Series:
    """The issue-day value carried forward."""
    return s.rename(f"pers_h{h}")


def recurrence_lag(h: int, rotation: int = ROTATION_DAYS, rotations_back: int = 0) -> int:
    """Days before the issue date of the recurrence analogue for lead `h`.

    `rotations_back=0` is the most recent usable rotation, 1 the one before it, etc.
    The lag is always ≥ 0 (never a future value).
    """
    k = math.ceil(h / rotation) + rotations_back
    return rotation * k - h


def recurrence(s: pd.Series, h: int, rotation: int = ROTATION_DAYS, n_rotations: int = 1) -> pd.Series:
    """Mean of the analogue values one rotation apart, starting from the latest usable one.

    n_rotations=1 is the classic 27-day recurrence forecast; 2 averages the last two
    rotations, which damps single-rotation noise at the cost of a longer lookback.
    """
    parts = [s.shift(recurrence_lag(h, rotation, r)) for r in range(n_rotations)]
    return pd.concat(parts, axis=1).mean(axis=1).rename(f"rec{n_rotations}_h{h}")


def target(s: pd.Series, h: int) -> pd.Series:
    """The verifying value for lead `h`, aligned on the issue date."""
    return s.shift(-h).rename(f"y_h{h}")


BASELINES = {
    "climatology": lambda s, h, **kw: climatology(s, h, kw.get("train_end")),
    "persistence": lambda s, h, **kw: persistence(s, h),
    "recurrence27": lambda s, h, **kw: recurrence(s, h, kw.get("rotation", ROTATION_DAYS), 1),
    "recurrence2x27": lambda s, h, **kw: recurrence(s, h, kw.get("rotation", ROTATION_DAYS), 2),
}
