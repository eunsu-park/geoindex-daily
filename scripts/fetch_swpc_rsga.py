"""Collect SWPC's daily Report of Solar-Geophysical Activity (RSGA) from NCEI and parse its Ap lines.

The RSGA (22:00 UTC daily, NCEI archive 1966-) carries the 3-day "Predicted Afr/Ap", the comparator
for one-day-ahead daily Ap (work order 2026-10-23, Task 3), plus the estimated Ap of the issue day and
the observed Ap of the day before. Raw files are mirrored under
$SOLARIS_ARCHIVE_ROOT/space_weather/swpc/rsga/YYYY/ (existing non-empty files are kept). A day that NCEI
lacks or serves empty is taken from the SWPC FTP warehouse tarball of its year when present (byte-identical
where both exist) and listed in rsga/FILLED_FROM_WAREHOUSE.txt. The parsed long table goes to
$GEOINDEX_DAILY_DATA/swpc/rsga_ap.parquet; when a day has more than one issuance (a late one after
midnight), the latest of the day is kept — the regular 22:00 UTC report.

    python scripts/fetch_swpc_rsga.py                                   # 2010-09-01 .. yesterday
    python scripts/fetch_swpc_rsga.py --start 2022-01-01 --end 2025-12-31
    python scripts/fetch_swpc_rsga.py --parse-only                      # rebuild the table from the mirror
"""
import argparse
import io
import os
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.swpc import URL_RSGA, parse_rsga  # noqa: E402

URL_WAREHOUSE = "ftp://ftp.swpc.noaa.gov/pub/warehouse/{y:04d}/{y:04d}_RSGA.tar.gz"


def archive_dir() -> Path:
    root = os.environ.get("SOLARIS_ARCHIVE_ROOT")
    if not root:
        raise SystemExit("SOLARIS_ARCHIVE_ROOT is not set (source ~/.solaris_env)")
    return Path(root) / "space_weather" / "swpc" / "rsga"


def fetch_one(day: pd.Timestamp, dest: Path, session: requests.Session) -> str:
    """Download one day's RSGA into dest unless present; returns 'exists', 'ok', 'missing' or 'error: …'."""
    path = dest / f"{day.year:04d}" / f"{day:%Y%m%d}RSGA.txt"
    if path.exists() and path.stat().st_size > 0:
        return "exists"
    url = URL_RSGA.format(y=day.year, m=day.month, d=day.day)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return "missing"
            r.raise_for_status()
            if not r.content.strip():
                return "empty"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return "ok"
        except requests.RequestException as e:
            if attempt == 2:
                return f"error: {e}"
            time.sleep(5 * (attempt + 1))
    return "error"


def fill_from_warehouse(days, dest: Path) -> list[str]:
    """Write the RSGA of each given day from the SWPC warehouse tarball of its year, if it is there."""
    filled = []
    for year in sorted({d.year for d in days}):
        try:
            blob = urllib.request.urlopen(URL_WAREHOUSE.format(y=year), timeout=120).read()
        except OSError as e:
            print(f"warehouse {year}: {e}")
            continue
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
            for d in (d for d in days if d.year == year):
                m = members.get(f"{d:%Y%m%d}RSGA.txt")
                data = tar.extractfile(m).read() if m else b""
                if data.strip():
                    path = dest / f"{year:04d}" / m.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    filled.append(f"{d:%Y-%m-%d}")
    return filled


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2010-09-01")
    p.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--parse-only", action="store_true")
    args = p.parse_args()
    dest = archive_dir()
    days = pd.date_range(args.start, args.end, freq="D")

    if not args.parse_only:
        with requests.Session() as s, ThreadPoolExecutor(args.workers) as pool:
            status = list(pool.map(lambda d: fetch_one(d, dest, s), days))
        st = pd.Series(status, index=days)
        print(st.str.split(":").str[0].value_counts().to_string())
        bad = st[~st.isin(["ok", "exists"])]
        if len(bad):
            print("not from NCEI:", ", ".join(f"{d:%Y-%m-%d} ({v})" for d, v in bad.items()))
            for d in bad.index:                 # drop empty files kept by an earlier run
                f = dest / f"{d.year:04d}" / f"{d:%Y%m%d}RSGA.txt"
                if f.exists() and f.stat().st_size == 0:
                    f.unlink()
            filled = fill_from_warehouse(list(bad.index), dest)
            if filled:
                log = dest / "FILLED_FROM_WAREHOUSE.txt"
                old = set(log.read_text().split()) if log.exists() else set()
                log.write_text("\n".join(sorted(old | set(filled))) + "\n")
            print(f"filled from the SWPC warehouse: {len(filled)} ({', '.join(filled)})")

    parts, unparsed = [], []
    for d in days:
        f = dest / f"{d.year:04d}" / f"{d:%Y%m%d}RSGA.txt"
        if not f.exists():
            continue
        try:
            parts.append(parse_rsga(f.read_text(errors="replace")))
        except Exception as e:  # noqa: BLE001 — record and continue
            unparsed.append(f"{f.name}: {type(e).__name__}")
    df = pd.concat(parts, ignore_index=True)
    last = df.groupby("issue_date").issue_time.transform("max")   # the regular 22:00 report of the day
    n_late = int((df.issue_time != last).sum())
    df = df[df.issue_time == last].drop_duplicates().reset_index(drop=True)   # one report filed under two names
    out = default_data_dir() / "swpc" / "rsga_ap.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    pred1 = df[(df.kind == "predicted") & (df.lead == 1)]
    print(f"wrote {out}: {df.issue_date.nunique()} issues, {len(df)} rows; "
          f"day-1 predicted Ap on {pred1.ap.notna().sum()} issues; unparsed {len(unparsed)}; "
          f"{n_late} rows of earlier same-day issuances dropped")
    if unparsed:
        print("unparsed:", "; ".join(unparsed[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
