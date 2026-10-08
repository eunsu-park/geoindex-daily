"""Samples, splits and uncertainty for the lead-1 track (daily Ap one day ahead).

A sample issued on day t has inputs over days t−L+1 .. t and target Ap(t+1). Every arm and
every input length L is scored on the SAME issue days (`common_issue_days`), so that the
curve over L and the differences between arms are paired comparisons.

Scores are MAE and CC in Ap units; the uncertainty of a difference between two forecasts
on the same days is a monthly block bootstrap over issue dates.
"""
import numpy as np
import pandas as pd

from .metrics import corr, mae, score
from .slots import TARGET_SOURCES, day_completeness


def feature_windows(df: pd.DataFrame, cols: list[str], L: int, issue_days: pd.DatetimeIndex,
                    target_col: str = "ap", lead: int = 1):
    """Inputs `X (N, L, F)` over days t−L+1..t and targets `y (N,)` = target_col(t+lead).

    `issue_days` must be a subset of df.index with the window and target available (see
    `common_issue_days`); nothing is dropped here.
    """
    pos = df.index.get_indexer(issue_days)
    assert (pos >= 0).all(), "issue day missing from the feature table"
    F = df[cols].to_numpy(dtype=float)
    X = np.stack([F[p - L + 1: p + 1] for p in pos])
    y = df[target_col].to_numpy(dtype=float)[pos + lead]
    return X, y


def image_complete_days(slot_table: pd.DataFrame, sources=TARGET_SOURCES,
                        slot_hours=(0, 6, 12, 18), window_days: int = 7) -> pd.DatetimeIndex:
    """Issue days whose last `window_days` days are complete in every image source."""
    comp = day_completeness(slot_table, sources, slot_hours)["all"]
    comp = comp.reindex(pd.date_range(comp.index.min(), comp.index.max(), freq="D"), fill_value=False)
    run = comp.astype(int).rolling(window_days, min_periods=window_days).sum()
    return pd.DatetimeIndex(run.index[run == window_days])


def common_issue_days(df: pd.DataFrame, all_cols: list[str], history_days: int, lead: int,
                      period: tuple[str, str], image_days: pd.DatetimeIndex | None = None,
                      target_col: str = "ap") -> pd.DatetimeIndex:
    """Issue days with `history_days` of gap-free features, a target, and (optionally) images."""
    ok = df[all_cols].notna().all(axis=1).astype(int)
    hist = ok.rolling(history_days, min_periods=history_days).sum() == history_days
    tgt = df[target_col].shift(-lead).notna()
    days = df.index[(hist & tgt).to_numpy()]
    lo, hi = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    days = days[(days >= lo) & (days <= hi)]
    if image_days is not None:
        days = days.intersection(image_days)
    return pd.DatetimeIndex(days, name="issue_date")


def split_masks(issue_days: pd.DatetimeIndex, spec: dict) -> dict[str, np.ndarray]:
    """`{train, val, test}` boolean masks from a date-range spec or a year-list spec."""
    if "test_years" in spec:
        yrs = issue_days.year
        test = np.isin(yrs, spec["test_years"])
        val = np.isin(yrs, spec["val_years"])
        return {"train": ~(test | val), "val": val, "test": test}
    out = {}
    for name in ("train", "val", "test"):
        lo, hi = pd.Timestamp(spec[name][0]), pd.Timestamp(spec[name][1])
        out[name] = np.asarray((issue_days >= lo) & (issue_days <= hi))
    return out


def ffill_features(df: pd.DataFrame, cols: list[str], limit: int) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].ffill(limit=limit)
    return out


def score_lead1(y: np.ndarray, p: np.ndarray, dates: pd.DatetimeIndex, threshold: float = 30.0) -> dict:
    s = pd.Series(y, index=dates)
    return score(s, pd.Series(p, index=dates), threshold)


def block_bootstrap_diff(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, dates: pd.DatetimeIndex,
                         n: int = 1000, seed: int = 0) -> dict:
    """Monthly block bootstrap of MAE(pa) − MAE(pb) and CC(pa) − CC(pb) on the same days.

    Returns the point estimate and the 2.5 / 97.5 percentiles for both statistics.
    """
    rng = np.random.default_rng(seed)
    months = pd.PeriodIndex(dates, freq="M")
    blocks = [np.flatnonzero(months == m) for m in months.unique()]
    d_mae, d_cc = [], []
    for _ in range(n):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        yy, a, b = y[idx], pa[idx], pb[idx]
        d_mae.append(np.abs(yy - a).mean() - np.abs(yy - b).mean())
        d_cc.append(np.corrcoef(yy, a)[0, 1] - np.corrcoef(yy, b)[0, 1])
    d_mae, d_cc = np.array(d_mae), np.array(d_cc)
    sy = pd.Series(y, index=dates)
    return {
        "d_mae": mae(sy, pd.Series(pa, index=dates)) - mae(sy, pd.Series(pb, index=dates)),
        "d_mae_lo": float(np.percentile(d_mae, 2.5)), "d_mae_hi": float(np.percentile(d_mae, 97.5)),
        "d_corr": corr(sy, pd.Series(pa, index=dates)) - corr(sy, pd.Series(pb, index=dates)),
        "d_corr_lo": float(np.percentile(d_cc, 2.5)), "d_corr_hi": float(np.percentile(d_cc, 97.5)),
    }
