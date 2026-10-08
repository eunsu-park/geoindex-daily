"""Daily feature table for the lead-1 track: Ap and friends plus GOES X-ray and solar wind.

Joins the daily index table (`daily_index.parquet`: Ap, Kp, F10.7, SN, Dst) with two
daily aggregates computed from the `space_weather` database:

- GOES XRS-B (0.1–0.8 nm) 1-min flux, all satellites with a good flag: the daily mean and
  the daily maximum, both as log10(W m⁻²). Overlapping satellites agree to a few per cent,
  so they are pooled rather than chosen.
- OMNI hourly solar wind: daily mean speed, proton density, |B| and Bz (GSM), and the
  daily minimum Bz. OMNI fill values (9999 km/s, 999.9) are masked before averaging.

Each aggregate carries its valid-sample count so gaps can be seen; nothing is filled here.
"""
from pathlib import Path

import pandas as pd

XRAY_SQL = """
select datetime::date as date,
       avg(xrs_b_flux_w_m2) as xray_mean,
       max(xrs_b_flux_w_m2) as xray_max,
       count(*)             as n_xray_min
from goes_xrs
where coalesce(xrs_b_flag, 0) = 0 and xrs_b_flux_w_m2 > 0
  and datetime >= %s and datetime < %s
group by 1 order by 1
"""

SW_SQL = """
select datetime::date as date,
       avg(case when plasma_flow_speed_km_s < 9000 then plasma_flow_speed_km_s end) as sw_v,
       avg(case when proton_density_n_cm3 < 999 then proton_density_n_cm3 end)       as sw_np,
       avg(case when b_field_magnitude_avg_nt < 999 then b_field_magnitude_avg_nt end) as sw_bt,
       avg(case when bz_gsm_nt < 999 then bz_gsm_nt end)                              as sw_bz_mean,
       min(case when bz_gsm_nt < 999 then bz_gsm_nt end)                              as sw_bz_min,
       count(case when plasma_flow_speed_km_s < 9000 then 1 end)                      as n_sw_hours
from omni_low_resolution
where datetime >= %s and datetime < %s
group by 1 order by 1
"""

XRAY_COLUMNS = ["xray_mean_log", "xray_max_log", "n_xray_min"]
SW_COLUMNS = ["sw_v", "sw_np", "sw_bt", "sw_bz_mean", "sw_bz_min", "n_sw_hours"]


def _daily_frame(conn, sql, start, end) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql, (start, end))
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_daily_xray(conn, start: str, end: str) -> pd.DataFrame:
    """Columns XRAY_COLUMNS indexed by UT day; end is exclusive."""
    import numpy as np
    df = _daily_frame(conn, XRAY_SQL, start, end)
    out = pd.DataFrame(index=df.index)
    out["xray_mean_log"] = np.log10(df["xray_mean"].astype(float))
    out["xray_max_log"] = np.log10(df["xray_max"].astype(float))
    out["n_xray_min"] = df["n_xray_min"].astype(int)
    return out


def load_daily_solarwind(conn, start: str, end: str) -> pd.DataFrame:
    """Columns SW_COLUMNS indexed by UT day; end is exclusive."""
    df = _daily_frame(conn, SW_SQL, start, end)
    for c in SW_COLUMNS[:-1]:
        df[c] = df[c].astype(float)
    df["n_sw_hours"] = df["n_sw_hours"].astype(int)
    return df[SW_COLUMNS]


def build_daily_features(daily_index_path: Path, conn, out_path: Path,
                         start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Join the daily index with the X-ray and solar-wind aggregates on a full daily grid."""
    daily = pd.read_parquet(daily_index_path)
    daily.index = pd.to_datetime(daily.index)
    start = start or daily.index.min().strftime("%Y-%m-%d")
    end = end or (daily.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    grid = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D", name="date")
    out = daily.reindex(grid)
    out = out.join(load_daily_xray(conn, start, end)).join(load_daily_solarwind(conn, start, end))
    out["n_xray_min"] = out["n_xray_min"].fillna(0).astype(int)
    out["n_sw_hours"] = out["n_sw_hours"].fillna(0).astype(int)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    return out


def load_daily_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df
