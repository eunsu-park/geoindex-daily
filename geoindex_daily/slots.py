"""Per-day image slot table from the `solar_images` catalog tables.

For every UT day and slot hour (00/06/12/18 by default) and every image source, the
frame nearest to the slot time within a tolerance is recorded; a slot with no frame
inside the tolerance is kept with NaN so completeness can be counted. The table is
long-format (one row per day × slot × source) and is the single place the daily image
pipeline reads frame paths from. File paths are relative to the archive root
(`SOLARIS_ARCHIVE_ROOT`), exactly as the catalog stores them.

Sources (catalog query → frames):
    lasco_c2  SOHO/LASCO C2, Orange filter (98 % of C2 frames)
    aia193    SDO/AIA 193 Å lev1, quality 0
    aia211    SDO/AIA 211 Å lev1, quality 0
    hmi_m45   SDO/HMI 45 s magnetogram, quality 0
    surya     SuryaBench 13-channel NetCDF (AIA + HMI, registered), hourly
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

SLOT_HOURS_4 = (0, 6, 12, 18)
SLOT_HOURS_2 = (0, 12)
TARGET_SOURCES = ("lasco_c2", "aia193", "aia211", "hmi_m45")
SOURCES = {
    "lasco_c2": "select datetime, file_path from lasco where camera = 'c2' and filter = 'Orange'",
    "aia193": "select datetime, file_path from sdo where telescope = 'aia' and wavelength = 193 "
              "and coalesce(quality, 0) = 0",
    "aia211": "select datetime, file_path from sdo where telescope = 'aia' and wavelength = 211 "
              "and coalesce(quality, 0) = 0",
    "hmi_m45": "select datetime, file_path from sdo where telescope = 'hmi' and channel = 'm_45s' "
               "and coalesce(quality, 0) = 0",
    "surya": "select datetime, file_path from suryabench",
}
COLUMNS = ["date", "slot_hour", "source", "datetime", "file_path", "offset_min"]


def default_archive_root() -> Path:
    return Path(os.environ.get("SOLARIS_ARCHIVE_ROOT", Path.home() / "Archive"))


def load_catalog(conn, source: str) -> pd.DataFrame:
    """All frames of one source as a frame sorted by datetime: columns datetime, file_path."""
    cur = conn.cursor()
    cur.execute(SOURCES[source] + " order by datetime")
    df = pd.DataFrame(cur.fetchall(), columns=["datetime", "file_path"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def nearest_frames(catalog: pd.DataFrame, targets: pd.DatetimeIndex,
                   tol_min: float = 180.0) -> pd.DataFrame:
    """For each target time, the catalog frame nearest in time, NaN beyond `tol_min` minutes.

    Returns a frame aligned with `targets` with columns datetime, file_path, offset_min
    (frame time minus target time, in minutes).
    """
    out = pd.DataFrame({"datetime": pd.NaT, "file_path": None, "offset_min": np.nan},
                       index=range(len(targets)))
    if catalog.empty or len(targets) == 0:
        return out
    times = catalog["datetime"].to_numpy(dtype="datetime64[ns]")
    tgt = targets.to_numpy(dtype="datetime64[ns]")
    right = np.searchsorted(times, tgt, side="left")
    left = np.clip(right - 1, 0, len(times) - 1)
    right = np.clip(right, 0, len(times) - 1)
    d_left = (tgt - times[left]).astype("timedelta64[s]").astype(float) / 60.0
    d_right = (times[right] - tgt).astype("timedelta64[s]").astype(float) / 60.0
    pick_right = d_right < d_left
    idx = np.where(pick_right, right, left)
    offset = np.where(pick_right, d_right, -d_left)
    ok = np.abs(offset) <= tol_min
    out.loc[ok, "datetime"] = catalog["datetime"].to_numpy()[idx[ok]]
    out.loc[ok, "file_path"] = catalog["file_path"].to_numpy()[idx[ok]]
    out.loc[ok, "offset_min"] = offset[ok]
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out


def slot_targets(start: str, end: str, slot_hours=SLOT_HOURS_4) -> pd.DataFrame:
    """One row per (date, slot_hour) over [start, end] inclusive, with the slot timestamp."""
    days = pd.date_range(start, end, freq="D")
    rows = pd.DataFrame({"date": np.repeat(days, len(slot_hours)),
                         "slot_hour": np.tile(list(slot_hours), len(days))})
    rows["slot_time"] = rows["date"] + pd.to_timedelta(rows["slot_hour"], unit="h")
    return rows


def build_slot_table(conn, start: str, end: str, slot_hours=SLOT_HOURS_4,
                     tol_min: float = 180.0, sources=tuple(SOURCES)) -> pd.DataFrame:
    """Long table (COLUMNS) of the nearest frame per day × slot × source."""
    targets = slot_targets(start, end, slot_hours)
    parts = []
    for src in sources:
        cat = load_catalog(conn, src)
        near = nearest_frames(cat, pd.DatetimeIndex(targets["slot_time"]), tol_min)
        part = targets[["date", "slot_hour"]].copy()
        part["source"] = src
        part[["datetime", "file_path", "offset_min"]] = near[["datetime", "file_path", "offset_min"]].to_numpy()
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    table["datetime"] = pd.to_datetime(table["datetime"])
    table["offset_min"] = table["offset_min"].astype(float)
    return table[COLUMNS]


def check_files(table: pd.DataFrame, root: Path | None = None) -> pd.Series:
    """Boolean per row: the frame file exists under the archive root (NaN rows → False)."""
    root = root or default_archive_root()
    cache: dict[str, bool] = {}

    def exists(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return False
        if p not in cache:
            cache[p] = (root / p).exists()
        return cache[p]

    return table["file_path"].map(exists).astype(bool)


def day_completeness(table: pd.DataFrame, sources=TARGET_SOURCES,
                     slot_hours=SLOT_HOURS_4) -> pd.DataFrame:
    """Per date: one bool column per source (every slot in `slot_hours` filled) plus `all`."""
    sub = table[table["source"].isin(sources) & table["slot_hour"].isin(slot_hours)]
    filled = sub.assign(ok=sub["file_path"].notna())
    per = filled.pivot_table(index="date", columns="source", values="ok", aggfunc="all")
    per = per.reindex(columns=list(sources)).fillna(False).astype(bool)
    per["all"] = per[list(sources)].all(axis=1)
    return per


def window_counts(complete: pd.Series, lengths=range(1, 8)) -> pd.DataFrame:
    """Issue days whose last `L` days are all complete, for each L (index L, column n_windows).

    `complete` is a boolean series indexed by consecutive daily dates.
    """
    c = complete.astype(int)
    rows = []
    for L in lengths:
        run = c.rolling(L, min_periods=L).sum()
        rows.append({"L": L, "n_windows": int((run == L).sum())})
    return pd.DataFrame(rows).set_index("L")


def save_slot_table(table: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def load_slot_table(path: Path) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t["date"] = pd.to_datetime(t["date"])
    t["datetime"] = pd.to_datetime(t["datetime"])
    return t
