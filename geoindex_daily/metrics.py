"""Verification scores for daily-index forecasts.

Ap is heavy-tailed (median ≈ 6, 99.9th percentile ≈ 100), so a linear MAE rewards
forecasting the mean and hides storm skill, while a log score and the storm-day
contingency table show the opposite face. All are reported side by side; which one
decides "beats SWPC" is a design choice that must be fixed before the comparison.
"""
import numpy as np
import pandas as pd

STORM_THRESHOLD = 30.0  # daily Ap >= 30 ≈ Kp 4+ for most of the day


def _align(y: pd.Series, yhat: pd.Series) -> pd.DataFrame:
    return pd.concat({"y": y, "yhat": yhat}, axis=1).dropna()


def mae(y, yhat) -> float:
    m = _align(y, yhat)
    return float((m.y - m.yhat).abs().mean())


def rmse(y, yhat) -> float:
    m = _align(y, yhat)
    return float(np.sqrt(((m.y - m.yhat) ** 2).mean()))


def log_mae(y, yhat) -> float:
    """MAE in log1p space — the scale on which Ap errors are roughly homoscedastic."""
    m = _align(y, yhat)
    return float((np.log1p(m.y) - np.log1p(m.yhat)).abs().mean())


def corr(y, yhat) -> float:
    m = _align(y, yhat)
    if len(m) < 3 or m.yhat.std() == 0:
        return float("nan")
    return float(np.corrcoef(m.y, m.yhat)[0, 1])


def storm_scores(y, yhat, threshold: float = STORM_THRESHOLD) -> dict:
    """Contingency scores for the event `value >= threshold`: POD, FAR, CSI, and counts."""
    m = _align(y, yhat)
    obs, fc = m.y >= threshold, m.yhat >= threshold
    hits = int((obs & fc).sum())
    misses = int((obs & ~fc).sum())
    false = int((~obs & fc).sum())
    pod = hits / (hits + misses) if hits + misses else float("nan")
    far = false / (hits + false) if hits + false else float("nan")
    csi = hits / (hits + misses + false) if hits + misses + false else float("nan")
    return {"pod": pod, "far": far, "csi": csi, "n_events": hits + misses, "n_forecast": hits + false}


def score(y, yhat, threshold: float = STORM_THRESHOLD) -> dict:
    """All scores for one (target, forecast) pair, as a flat dict."""
    out = {"n": len(_align(y, yhat)), "mae": mae(y, yhat), "rmse": rmse(y, yhat),
           "log_mae": log_mae(y, yhat), "corr": corr(y, yhat)}
    out.update(storm_scores(y, yhat, threshold))
    return out
