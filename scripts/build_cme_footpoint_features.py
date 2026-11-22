"""Optional CME inputs (Model A) and Solar-MACH footpoint features (Model B) per issue day, as of the issue time.

Co-authors' work order of 2026-11-22 (vault inbox/20261122) and their reply of the same day. Issue day t is issued at
T = 00 UT of t + 1 (as every ap3h model). Evaluation level 1 (retrospective): DONKI's present records are used, with
each analysis admitted only from its `submissionTime`; whether DONKI rewrites that time on an edit cannot be checked.

CME slots. A CME is eligible at T when it started in (T − 5 d, T] and at least one of its DONKI analyses with speed,
latitude and longitude has submissionTime ≤ T; its values are those of the latest such analysis (DONKI's
`isMostAccurate` flag is set afterwards and is not used). No restriction to Earth-directed CMEs. The eligible CMEs are
ordered by start time, latest first, and the first three fill slots 1–3 (speed km/s, latitude and longitude in
Stonyhurst degrees as DONKI gives them, time since onset h, analysis submission time).

CME status at T (one code per issue; the model sees only the slot masks):
* available            — at least one eligible CME;
* no_cme               — no CME in DONKI (present catalogue) started in the window;
* not_yet_analysed     — CMEs started in the window, but their analysis came after T (within 30 days of the start);
* no_realtime_record   — CMEs in the window whose analyses were all entered more than 30 days after the start
                         (DONKI's backfill of 2010–2013): unavailable in real time;
* incomplete_analysis  — CMEs in the window with analyses before T that lack speed or direction.
The order above is the precedence when the window holds CMEs of several kinds.

Footpoint. Earth's magnetic footpoint at 1 R_sun on an ideal Parker spiral for a fixed solar-wind speed v (300, 400,
600 km/s), as Solar-MACH 0.5.7 computes it (`backmapping`): φ_fp = φ_E + Ω(β)·(r − R_sun)·cos β / v, with φ_E, β, r the
Carrington longitude (observer = Sun), latitude and distance of Earth at T and Ω(β) = 14.50 − 2.87 sin²β deg/day.
Earth's position comes from astropy's built-in ephemeris (offline), not JPL Horizons; scripts/check_footpoint_solarmach.py
compares a sample of dates with the package. Encoded as sin φ_fp, cos φ_fp (radians), six values.

    python scripts/build_cme_footpoint_features.py            # → $GEOINDEX_DAILY_DATA/events/ap3h_cme_footpoint.parquet
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

WINDOW = pd.Timedelta(days=5)
SLOTS = 3
BACKFILL = pd.Timedelta(days=30)
SPEEDS = (300, 400, 600)
STATUS_ORDER = ["available", "not_yet_analysed", "no_realtime_record", "incomplete_analysis", "no_cme"]


def footpoint_longitudes(times: pd.DatetimeIndex, speeds=SPEEDS) -> pd.DataFrame:
    """Earth's Carrington longitude, latitude, distance and footpoint longitude (deg, 0–360) per speed."""
    import astropy.units as u
    from astropy.constants import R_sun
    from sunpy.coordinates import frames, get_earth
    e = get_earth(times.to_pydatetime()).transform_to(frames.HeliographicCarrington(observer="Sun", obstime=times.to_pydatetime()))
    lon, lat, r = e.lon.to_value(u.deg), e.lat.to(u.rad), e.radius.to(u.km)
    omega = (14.50 - 2.87 * np.sin(lat) ** 2) * u.deg / u.day
    out = {"earth_carr_lon": lon, "earth_lat": lat.to_value(u.deg), "earth_r_au": e.radius.to_value(u.AU)}
    for v in speeds:
        alpha = (omega * (r - R_sun.to(u.km)) * np.cos(lat) / (v * u.km / u.s)).to_value(u.deg)
        out[f"fp{v}_lon"] = np.mod(lon + alpha, 360.0)
    return pd.DataFrame(out, index=times)


def load_donki(raw: Path) -> pd.DataFrame:
    """One row per (CME, analysis): start, analysis submission, speed / lat / lon (None where missing)."""
    rows = []
    for f in sorted(glob.glob(str(raw / "*.json"))):
        for e in json.loads(Path(f).read_text() or "[]"):
            start = pd.to_datetime(e["startTime"], utc=True).tz_localize(None)
            for a in e.get("cmeAnalyses") or [None]:
                a = a or {}
                sub = a.get("submissionTime")
                rows.append({"cme_id": e["activityID"], "start": start,
                             "submitted": pd.to_datetime(sub, utc=True).tz_localize(None) if sub else pd.NaT,
                             "speed": a.get("speed"), "lat": a.get("latitude"), "lon": a.get("longitude")})
    d = pd.DataFrame(rows).drop_duplicates()
    d["complete"] = d[["speed", "lat", "lon"]].notna().all(axis=1) & d.submitted.notna()
    return d.sort_values(["start", "submitted"]).reset_index(drop=True)


def cme_state(d: pd.DataFrame, T: pd.Timestamp) -> tuple[str, list[dict]]:
    w = d[(d.start > T - WINDOW) & (d.start <= T)]
    if w.empty:
        return "no_cme", []
    ok = w[w.complete & (w.submitted <= T)]
    if len(ok):
        last = ok.sort_values("submitted").groupby("cme_id").tail(1).sort_values("start", ascending=False).head(SLOTS)
        return "available", last.to_dict("records")
    kinds = set()
    for _, g in w.groupby("cme_id"):
        if (g.complete & (g.submitted > T) & (g.submitted - g.start <= BACKFILL)).any():
            kinds.add("not_yet_analysed")
        elif g.complete.any() and (g[g.complete].submitted - g.start > BACKFILL).all():
            kinds.add("no_realtime_record")
        else:
            kinds.add("incomplete_analysis")
    return next(s for s in STATUS_ORDER if s in kinds), []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2010-08-25", help="first issue day t (default: a week before split A)")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = os.environ.get("SOLARIS_ARCHIVE_ROOT")
    if not root:
        raise SystemExit("SOLARIS_ARCHIVE_ROOT is not set (source ~/.solaris_env)")
    d = load_donki(Path(root) / "space_weather" / "ccmc" / "donki_cme")
    days = pd.date_range(args.start, args.end, freq="D")
    issue = days + pd.Timedelta(days=1)
    rows = []
    for t, T in zip(days, issue):
        status, slots = cme_state(d, T)
        r = {"issue_day": t, "issue_time": T, "cme_status": status, "n_cme": len(slots)}
        for k in range(SLOTS):
            s = slots[k] if k < len(slots) else None
            r.update({f"cme{k + 1}_mask": int(s is not None),
                      f"cme{k + 1}_id": s["cme_id"] if s else None,
                      f"cme{k + 1}_speed": float(s["speed"]) if s else np.nan,
                      f"cme{k + 1}_lat": float(s["lat"]) if s else np.nan,
                      f"cme{k + 1}_lon": float(s["lon"]) if s else np.nan,
                      f"cme{k + 1}_elapsed_h": (T - s["start"]) / pd.Timedelta(hours=1) if s else np.nan,
                      f"cme{k + 1}_submitted": s["submitted"] if s else pd.NaT})
        rows.append(r)
    out = pd.DataFrame(rows)
    fp = footpoint_longitudes(pd.DatetimeIndex(issue))
    for c in fp.columns:
        out[c] = fp[c].to_numpy()
    for v in SPEEDS:
        rad = np.deg2rad(out[f"fp{v}_lon"])
        out[f"fp{v}_sin"], out[f"fp{v}_cos"] = np.sin(rad), np.cos(rad)

    # QC: no admitted analysis after the issue time; slots in start order; time zone naive UTC throughout
    for k in range(SLOTS):
        m = out[f"cme{k + 1}_mask"] == 1
        assert (out.loc[m, f"cme{k + 1}_submitted"] <= out.loc[m, "issue_time"]).all()
        assert (out.loc[m, f"cme{k + 1}_elapsed_h"].between(0, 120)).all()
    assert (out.cme2_mask <= out.cme1_mask).all() and (out.cme3_mask <= out.cme2_mask).all()
    assert out[[f"fp{v}_lon" for v in SPEEDS]].notna().all().all()

    path = args.out or default_data_dir() / "events" / "ap3h_cme_footpoint.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path}: {len(out)} issue days {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}")
    split = pd.cut(out.issue_day, pd.to_datetime(["2000-01-01", "2020-01-01", "2022-01-01", "2030-01-01"]),
                   labels=["train", "val", "test"], right=False)
    print(pd.crosstab(out.cme_status, split, margins=True).to_string())
    print("CMEs per issue (available):", out[out.n_cme > 0].n_cme.value_counts().sort_index().to_dict())
    print("available share by year:", out.groupby(out.issue_day.dt.year).apply(lambda g: round((g.n_cme > 0).mean(), 2)).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
