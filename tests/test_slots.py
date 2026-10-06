import numpy as np
import pandas as pd

from geoindex_daily.slots import (
    day_completeness, nearest_frames, slot_targets, window_counts,
)


def _catalog(times):
    t = pd.to_datetime(times)
    return pd.DataFrame({"datetime": t, "file_path": [f"f/{x.strftime('%Y%m%d_%H%M')}" for x in t]})


def test_nearest_picks_closest_side_and_signs_offset():
    cat = _catalog(["2020-01-01 05:00", "2020-01-01 06:30", "2020-01-01 13:00"])
    tg = pd.DatetimeIndex(pd.to_datetime(["2020-01-01 06:00", "2020-01-01 12:00"]))
    near = nearest_frames(cat, tg, tol_min=180)
    assert near.file_path.tolist() == ["f/20200101_0630", "f/20200101_1300"]
    assert near.offset_min.tolist() == [30.0, 60.0]


def test_nearest_beyond_tolerance_is_nan_and_edges_are_safe():
    cat = _catalog(["2020-01-01 00:00"])
    tg = pd.DatetimeIndex(pd.to_datetime(["2019-12-31 18:00", "2020-01-01 04:00", "2020-01-02 00:00"]))
    near = nearest_frames(cat, tg, tol_min=180)
    assert near.file_path.tolist() == [None, None, None]
    assert near.offset_min.isna().all()
    near = nearest_frames(cat, tg, tol_min=360)
    assert near.file_path.iloc[0] == "f/20200101_0000" and near.offset_min.iloc[0] == 360.0
    assert near.offset_min.iloc[1] == -240.0


def test_slot_targets_shape():
    t = slot_targets("2020-01-01", "2020-01-03")
    assert len(t) == 12 and t.slot_time.iloc[5] == pd.Timestamp("2020-01-02 06:00")


def test_day_completeness_and_window_counts():
    days = pd.date_range("2020-01-01", periods=5, freq="D")
    rows = []
    for d in days:
        for h in (0, 6, 12, 18):
            for src in ("a", "b"):
                ok = not (src == "b" and d.day == 3 and h == 18)  # b misses one slot on day 3
                rows.append({"date": d, "slot_hour": h, "source": src,
                             "datetime": d if ok else pd.NaT, "file_path": "x" if ok else None,
                             "offset_min": 0.0 if ok else np.nan})
    table = pd.DataFrame(rows)
    comp = day_completeness(table, ("a", "b"), (0, 6, 12, 18))
    assert comp["a"].all() and comp["b"].sum() == 4 and comp["all"].sum() == 4
    comp2 = day_completeness(table, ("a", "b"), (0, 12))
    assert comp2["all"].all()
    wc = window_counts(comp["all"], lengths=[1, 2, 3])
    # complete: T T F T T → L=1: 4, L=2: 2 (days 1-2, 4-5), L=3: 0
    assert wc.n_windows.tolist() == [4, 2, 0]
