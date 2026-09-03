"""SWPC operational forecasts of daily Ap / F10.7, for the comparison the paper needs.

Three sources, found 2026-09-03 (see docs in the geoindex hub, geoindex-daily/overview.md):

1. **45-day forecast, JSON, NCEI archive** — daily issuances since 2026-03 only:
   `https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/45_day_forecast/YYYY/MM/45_day_forecast_YYYYMMDD.json`
2. **45-day forecast, text, live** — today's issuance only (no public archive before 2026-03;
   the Wayback Machine holds nothing usable either):
   `https://services.swpc.noaa.gov/text/45-day-forecast.txt`
3. **27-day outlook inside the weekly PRF** ("Preliminary Report and Forecast of Solar
   Geophysical Data"), every Monday, PDF, archived 1997–present:
   `https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/weekly_reports/PRFs_of_SGD/YYYY/MM/prfNNNN.pdf`
   Page 3, "Twenty-seven Day Outlook": date, F10.7, Ap, largest Kp — two columns of
   14 + 13 days starting on the issue Monday. This is the only SWPC daily-Ap forecast
   with a multi-year archive, so it is the historical comparator (leads 1–27 d, weekly).

Every parser returns a long table: `issue_date, target_date, lead, ap, f107` (Kp too for
the PRF), one row per forecast day.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

NCEI = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products"
URL_45DAY_JSON = NCEI + "/daily_reports/45_day_forecast/{y:04d}/{m:02d}/45_day_forecast_{y:04d}{m:02d}{d:02d}.json"
URL_45DAY_TXT = "https://services.swpc.noaa.gov/text/45-day-forecast.txt"
URL_PRF_DIR = NCEI + "/weekly_reports/PRFs_of_SGD/{y:04d}/{m:02d}/"

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _long(issue: pd.Timestamp, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "issue_date", pd.Timestamp(issue).normalize())
    df["lead"] = (df["target_date"] - df["issue_date"]).dt.days
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


# --- PRF 27-day outlook ---------------------------------------------------------------

_PRF_HEADER = re.compile(r"SWPC PRF (\d+)\s+(\d{2}) ([A-Z][a-z]+) (\d{4})")
_ENTRY = re.compile(r"(\d{2})(?:\s+([A-Z][a-z]{2}))?\s+(\d+)\s+(\d+)\s+(\d+)")


def pdf_text(path: Path) -> str:
    """Layout-preserving text of a PDF (pdftotext if available, else pypdf's layout mode)."""
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                              capture_output=True, text=True).stdout
    from pypdf import PdfReader
    return "\n".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(str(path)).pages)


def parse_prf_outlook(text: str) -> pd.DataFrame:
    """The 'Twenty-seven Day Outlook' table of one PRF → long table (f107, ap, kp).

    The table has two side-by-side columns (14 + 13 days). Month labels appear only on
    the first day of the table and on month changes; the issue date (from the page
    header) fixes the year, with a wrap for outlooks that cross New Year.
    """
    m = _PRF_HEADER.search(text)
    if not m:
        raise ValueError("no 'SWPC PRF NNNN dd Month yyyy' header found")
    issue = pd.Timestamp(f"{m.group(4)}-{m.group(3)[:3]}-{m.group(2)}")
    # The title also appears in the table of contents of some issues; take the
    # occurrence that is actually followed by the table header.
    start = -1
    for m_ in re.finditer("Twenty-seven Day Outlook", text):
        if "Radio Flux" in text[m_.start(): m_.start() + 800]:
            start = m_.start()
            break
    if start < 0:
        raise ValueError("no 'Twenty-seven Day Outlook' table found")
    # The table ends at the next page header ("N   SWPC PRF ..."): the following page
    # holds the Energetic Events / Flare List, whose rows look just like outlook rows.
    # A header may also sit between the title and the table (page break), so only a
    # header seen after at least one table row ends the scan.
    left, right = [], []
    for line in text[start:].splitlines()[1:]:
        if "SWPC PRF" in line and left:
            break
        found = list(_ENTRY.finditer(line))
        if not found:
            continue
        left.append(found[0].groups())
        if len(found) > 1:
            right.append(found[1].groups())
    entries = left + right
    if not 26 <= len(entries) <= 27:  # a few issues list 26 days
        raise ValueError(f"{len(entries)} outlook rows parsed (expected 27)")

    # Month/year tracking starts from the issue date, so a late-December issue whose
    # table opens with "01 Jan" rolls the year forward.
    rows, month, year, prev_day = [], issue.month, issue.year, None
    for day, mon, f107, ap, kp in entries:
        day = int(day)
        if mon:
            new_month = MONTHS[mon]
            if month is not None and new_month < month:
                year += 1
            month = new_month
        elif prev_day is not None and day < prev_day:  # month rolled without a label
            month = month % 12 + 1
            if month == 1:
                year += 1
        rows.append({"target_date": pd.Timestamp(year=year, month=month, day=day),
                     "f107": float(f107), "ap": float(ap), "kp": float(kp)})
        prev_day = day
    return _long(issue, rows)


def prf_index(year: int, month: int, fetch) -> list[str]:
    """PRF file names listed in the NCEI month directory (`fetch(url) -> html`)."""
    html = fetch(URL_PRF_DIR.format(y=year, m=month))
    return sorted(set(re.findall(r'href="(prf\d+\.pdf)"', html)))
