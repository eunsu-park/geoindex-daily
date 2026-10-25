"""SWPC operational forecasts of daily Ap / F10.7, for the comparison the paper needs.

Sources (surveyed 2026-09-03):

1. **27-day outlook inside the weekly PRF** ("Preliminary Report and Forecast of Solar
   Geophysical Data"), every Monday, archived by NCEI 1997–present. The only SWPC daily-Ap
   forecast with a multi-year archive, hence the historical comparator (leads 1–26 d).
   Acquisition lives in **solaris-data** (`scripts/download_swpc_prf.py`, parser
   `core/swpc_prf.py`), which mirrors the PDFs and loads `space_weather.swpc_prf_outlook`.
   This repo reads that table: `load_prf_outlook` (and `scripts/load_swpc_prf.py` for the
   local parquet the eval scripts use).
2. **45-day forecast, JSON, NCEI archive** — daily issuances since 2026-03 only:
   `.../daily_reports/45_day_forecast/YYYY/MM/45_day_forecast_YYYYMMDD.json`.
3. **45-day forecast, text, live** — today's issuance only (no public archive before
   2026-03; the Wayback Machine holds nothing usable either):
   `https://services.swpc.noaa.gov/text/45-day-forecast.txt`.

The 45-day parsers are kept here for the prospective comparison; a collector for them is
deferred (decision 2026-09-03). Every function returns a long table:
`issue_date, target_date, lead, ap, f107` (kp too for the PRF), one row per forecast day.
"""
from __future__ import annotations

import json
import re

import pandas as pd

NCEI = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products"
URL_45DAY_JSON = NCEI + "/daily_reports/45_day_forecast/{y:04d}/{m:02d}/45_day_forecast_{y:04d}{m:02d}{d:02d}.json"
URL_45DAY_TXT = "https://services.swpc.noaa.gov/text/45-day-forecast.txt"
# Report of Solar-Geophysical Activity: daily, 22:00 UTC, 1966-; carries the 3-day "Predicted Afr/Ap"
URL_RSGA = NCEI + "/daily_reports/reports_solar_geophysical_activity/{y:04d}/{m:02d}/{y:04d}{m:02d}{d:02d}RSGA.txt"
PRF_TABLE = "swpc_prf_outlook"


def _long(issue: pd.Timestamp, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "issue_date", pd.Timestamp(issue).normalize())
    df["lead"] = (df["target_date"] - df["issue_date"]).dt.days
    return df


# --- PRF 27-day outlook: read the shared table -------------------------------------------

def load_prf_outlook(conn, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """`swpc_prf_outlook` rows (issue_date, target_date, lead, f107, ap, kp), issue-date bounded."""
    where, params = [], []
    if start:
        where.append("issue_date >= %s"); params.append(start)
    if end:
        where.append("issue_date <= %s"); params.append(end)
    sql = f"select issue_date, target_date, lead, f107, ap, kp from {PRF_TABLE}"
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by issue_date, target_date"
    cur = conn.cursor()
    cur.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=["issue_date", "target_date", "lead", "f107", "ap", "kp"])
    for c in ("issue_date", "target_date"):
        df[c] = pd.to_datetime(df[c])
    return df


# --- 45-day product -------------------------------------------------------------------

def parse_45day_json(text: str) -> pd.DataFrame:
    """NCEI/SWPC JSON issuance → long table (ap, f107)."""
    j = json.loads(text)
    issue = pd.Timestamp(j["issued"]).tz_localize(None)
    rows = [{"target_date": pd.Timestamp(r["time"]).tz_localize(None).normalize(),
             "ap": float(r["ap"]), "f107": float(r["f107"])} for r in j["data"]]
    return _long(issue, rows)


_TXT_PAIR = re.compile(r"(\d{2}[A-Z][a-z]{2}\d{2})\s+(\d+)")


def parse_45day_txt(text: str) -> pd.DataFrame:
    """`45-day-forecast.txt` → long table. Sections '45-DAY AP FORECAST' / '45-DAY F10.7 CM FLUX FORECAST'."""
    issue = pd.Timestamp(re.search(r":Issued:\s*(\d{4} \w{3} \d{2} \d{4}) UTC", text).group(1), )
    sections = re.split(r"\n(?=45-DAY )", text)
    ap, f107 = {}, {}
    for sec in sections:
        head = sec.splitlines()[0] if sec else ""
        target = ap if "AP" in head else f107 if "F10.7" in head else None
        if target is None:
            continue
        for d, v in _TXT_PAIR.findall(sec):
            target[pd.Timestamp(pd.to_datetime(d, format="%d%b%y"))] = float(v)
    rows = [{"target_date": t, "ap": ap.get(t), "f107": f107.get(t)} for t in sorted(set(ap) | set(f107))]
    return _long(issue, rows)


# --- RSGA (Report of Solar-Geophysical Activity) -------------------------------------------

_RSGA_PRED = re.compile(r"Predicted Afr/Ap\s+(\d{2} \w{3})-(\d{2} \w{3})\s+(.+)")
_RSGA_ONE = re.compile(r"(Observed|Estimated) Afr/Ap\s+(\d{2} \w{3})\s+(\S+)/\s*(\S+)")


def _num(v: str) -> float | None:
    """A number, or None for 'NA', 'Missing', '0NA', 'N60' and other non-numeric entries."""
    try:
        return float(v.strip())
    except ValueError:
        return None


def parse_rsga(text: str) -> pd.DataFrame:
    """One RSGA issuance → long table: issue_date, issue_time, target_date, lead, kind, afr, ap.

    kind is 'predicted' (leads 1-3), 'estimated' (the issue day, lead 0) or 'observed' (the day
    before, lead -1). Missing values ('NA') become None. Dates in the report carry no year; it is
    taken from the issue date, rolling over at the turn of the year.
    """
    m = re.search(r":Issued:\s*(\d{4} \w{3} \d{2}) (\d{4}) UTC", text)
    issue = pd.Timestamp(m.group(1))

    def day(dm: str) -> pd.Timestamp:
        t = pd.Timestamp(f"{dm} {issue.year}")
        if (t - issue).days > 180:          # e.g. 31 Dec observed in a January report
            t = t - pd.DateOffset(years=1)
        elif (issue - t).days > 180:        # e.g. 01 Jan predicted in a December report
            t = t + pd.DateOffset(years=1)
        return t

    rows = []
    for kind, dm, afr, ap in _RSGA_ONE.findall(text):
        rows.append({"target_date": day(dm), "kind": kind.lower(), "afr": _num(afr), "ap": _num(ap)})
    p = _RSGA_PRED.search(text)
    if p:
        start = day(p.group(1))
        for k, pair in enumerate(p.group(3).split("-")):
            afr, _, ap = pair.partition("/")
            rows.append({"target_date": start + pd.Timedelta(days=k), "kind": "predicted",
                         "afr": _num(afr), "ap": _num(ap)})
    cols = ["issue_date", "issue_time", "target_date", "lead", "kind", "afr", "ap"]
    if not rows:                            # truncated report without the Afr/Ap lines
        return pd.DataFrame(columns=cols)
    df = _long(issue, rows)
    df.insert(1, "issue_time", m.group(2))
    return df[cols]
