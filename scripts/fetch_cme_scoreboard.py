"""Collect the CCMC Earth CME Scoreboard (every listed CME, arrived or not) and build the CME catalog.

Work order 2026-10-23, §9 and the co-authors' answer to Q7 (2026-10-25): a CME is every entry of the
Earth CME Scoreboard, dated by its CME start time (the scoreboard's observed time); whether it arrived is
kept as a column, not used to select. Source: the scoreboard's web service
https://ccmc.gsfc.nasa.gov/CMESB-Earth/WS/get/predictions (no key), queried per calendar year with
skipNoArrivalObservedCMEs=false and closeOutCMEsOnly=false. Raw JSON is kept under
$SOLARIS_ARCHIVE_ROOT/space_weather/ccmc/cme_scoreboard/<year>.json; the catalog goes to
$GEOINDEX_DAILY_DATA/events/cme_scoreboard.parquet.

    python scripts/fetch_cme_scoreboard.py                       # 2010 .. this year
    python scripts/fetch_cme_scoreboard.py --years 2022 2023 2024 2025
    python scripts/fetch_cme_scoreboard.py --parse-only
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402

URL = ("https://ccmc.gsfc.nasa.gov/CMESB-Earth/WS/get/predictions?CMEtimeStart={y}-01-01&CMEtimeEnd={y}-12-31"
       "&skipNoArrivalObservedCMEs=false&closeOutCMEsOnly=false")
COLUMNS = ["cme_id", "cme_start_time", "cme_date", "arrived", "arrival_time", "max_kp", "dst_min",
           "n_predictions", "source", "source_url", "note"]


def catalog(entries: list[dict], source_url: str) -> pd.DataFrame:
    """Scoreboard entries → one row per CME (COLUMNS); times UTC, naive."""
    rows = []
    for e in entries:
        t = pd.Timestamp(e["observedTime"]).tz_localize(None) if e.get("observedTime") else \
            pd.Timestamp(e["cmeID"].split("-CME-")[0])
        arr = pd.Timestamp(e["arrivalTime"]).tz_localize(None) if e.get("arrivalTime") else pd.NaT
        rows.append({"cme_id": e["cmeID"], "cme_start_time": t, "cme_date": t.normalize(),
                     "arrived": (not e.get("noArrivalObserved")) and pd.notna(arr), "arrival_time": arr,
                     "max_kp": e.get("maxKP"), "dst_min": e.get("dstMin"),
                     "n_predictions": len(e.get("predictions") or []),
                     "source": "CCMC Earth CME Scoreboard", "source_url": source_url, "note": e.get("cmeNote")})
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs="*", type=int, default=list(range(2010, date.today().year + 1)))
    p.add_argument("--parse-only", action="store_true")
    args = p.parse_args()
    root = os.environ.get("SOLARIS_ARCHIVE_ROOT")
    if not root:
        raise SystemExit("SOLARIS_ARCHIVE_ROOT is not set (source ~/.solaris_env)")
    raw = Path(root) / "space_weather" / "ccmc" / "cme_scoreboard"
    raw.mkdir(parents=True, exist_ok=True)

    parts = []
    for y in args.years:
        f, url = raw / f"{y}.json", URL.format(y=y)
        if not args.parse_only:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        if f.exists():
            entries = json.loads(f.read_text())
            parts.append(catalog(entries, url))
            print(f"{y}: {len(entries)} CMEs")
    cat = pd.concat(parts, ignore_index=True).drop_duplicates("cme_id").sort_values("cme_start_time")
    out = default_data_dir() / "events" / "cme_scoreboard.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(out, index=False)
    print(f"wrote {out}: {len(cat)} CMEs, {int(cat.arrived.sum())} arrived "
          f"({cat.cme_date.min():%Y-%m-%d} .. {cat.cme_date.max():%Y-%m-%d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
