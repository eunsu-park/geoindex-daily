"""Daily index table (Ap, Kp, F10.7, SN, Dst) from the OMNI hourly table.

`space_weather.omni_low_resolution` carries the 3-hourly ap repeated on every hour of
its block, and the daily F10.7 / sunspot number repeated on all 24 hours. The mean of
the 24 hourly ap values is therefore exactly the official daily Ap (mean of the eight
3-hourly ap values); F10.7 and SN reduce to themselves. Kp is stored ×10 in OMNI and
is reported here as the daily maximum in Kp units; Dst as the daily minimum.

OMNI fill values (999, 99, 999.9, 99999) are masked before aggregation, and days with
fewer than 24 valid ap hours are kept but flagged via `n_ap_hours`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HOURLY_COLUMNS = ["datetime", "ap_index_nt", "kp_index", "f10_7_index_sfu",
                  "sunspot_number_r", "dst_index_nt"]
# OMNI low-resolution fill markers, applied as `value >= fill` (see omni2.text).
FILLS = {"ap_index_nt": 999, "kp_index": 99, "f10_7_index_sfu": 999.9,
         "sunspot_number_r": 999, "dst_index_nt": 99999}
DAILY_COLUMNS = ["ap", "kp", "f107", "sn", "dst", "n_ap_hours"]


def default_data_dir() -> Path:
    """Canonical local store for derived tables (cloud-synced, not on the NAS)."""
    import os
    return Path(os.environ.get("GEOINDEX_DAILY_DATA",
                               Path.home() / "Projects" / "GeoIndex" / "daily"))


def load_hourly_omni(conn, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Hourly OMNI rows (the columns in HOURLY_COLUMNS), fills masked to NaN."""
    where, params = [], []
    if start:
        where.append("datetime >= %s"); params.append(start)
    if end:
        where.append("datetime < %s"); params.append(end)
    sql = f"select {', '.join(HOURLY_COLUMNS)} from omni_low_resolution"
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by datetime"
    cur = conn.cursor()
    cur.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=HOURLY_COLUMNS)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return mask_fills(df)


def mask_fills(hourly: pd.DataFrame) -> pd.DataFrame:
    out = hourly.copy()
    for col, fill in FILLS.items():
        if col in out:
            out[col] = out[col].astype(float)
            out.loc[out[col] >= fill, col] = np.nan
    return out


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily table indexed by UT day (DatetimeIndex named `date`), columns DAILY_COLUMNS.

    The index is made contiguous (every day between first and last), so a day with no
    hourly rows appears as NaN with `n_ap_hours == 0` rather than being silently absent.
    """
    h = hourly.set_index("datetime")
    day = h.index.floor("D")
    g = h.groupby(day)
    daily = pd.DataFrame({
        "ap": g["ap_index_nt"].mean(),
        "kp": g["kp_index"].max() / 10.0,
        "f107": g["f10_7_index_sfu"].mean(),
        "sn": g["sunspot_number_r"].mean(),
        "dst": g["dst_index_nt"].min(),
        "n_ap_hours": g["ap_index_nt"].count(),
    })
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    daily = daily.reindex(full)
    daily["n_ap_hours"] = daily["n_ap_hours"].fillna(0).astype(int)
    return daily[DAILY_COLUMNS]


def build(out_path: Path | None = None, start: str | None = None, end: str | None = None) -> Path:
    """DB → daily parquet (default `<data dir>/daily_index.parquet`). Returns the path."""
    from .db import connect
    out_path = Path(out_path) if out_path else default_data_dir() / "daily_index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with connect("space_weather") as conn:
        daily = to_daily(load_hourly_omni(conn, start, end))
    daily.to_parquet(out_path)
    return out_path


def load_daily(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else default_data_dir() / "daily_index.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run scripts/build_daily_index.py first")
    return pd.read_parquet(path)
