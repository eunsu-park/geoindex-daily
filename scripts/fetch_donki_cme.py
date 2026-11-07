"""Collect the NASA DONKI CME list (2010 on) and flag the Earth-directed CMEs.

The CME-only model of the 3-hour ap plan (2026-11-06) selects issue days with an Earth-directed CME that
started in the 72 h before the issue time; the Earth CME Scoreboard (fetch_cme_scoreboard.py) starts in
2013-03 and is sparse before 2020, so DONKI is used instead (user, 2026-11-07). DONKI lists every CME that
the Moon to Mars Space Weather Analysis Office analysed, most of them not Earth-directed; a CME is flagged
Earth-directed when its most accurate cone analysis covers the Sun–Earth line (|longitude| ≤ half angle,
HEEQ) or when any ENLIL run of it predicts an Earth impact. Source: https://ccmc.gsfc.nasa.gov/DONKI/WS/get/CME
(no key; at most 60 days per request), queried per calendar month. Raw JSON is kept under
$SOLARIS_ARCHIVE_ROOT/space_weather/ccmc/donki_cme/<YYYY-MM>.json; the catalog goes to
$GEOINDEX_DAILY_DATA/events/donki_cme.parquet.

    python scripts/fetch_donki_cme.py                     # 2010-01 .. last month, missing months only
    python scripts/fetch_donki_cme.py --start 2025-01 --end 2025-12 --refresh
    python scripts/fetch_donki_cme.py --parse-only
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402

URL = "https://ccmc.gsfc.nasa.gov/DONKI/WS/get/CME?startDate={s:%Y-%m-%d}&endDate={e:%Y-%m-%d}"
COLUMNS = ["cme_id", "cme_start_time", "cme_date", "latitude", "longitude", "half_angle", "speed", "type",
           "technique", "cone_covers_earth", "enlil_earth", "earth_directed", "linked_ips", "linked_gst",
           "source_location", "link"]


def catalog(entries: list[dict]) -> pd.DataFrame:
    """DONKI CME entries → one row per CME (COLUMNS); times UTC, naive."""
    rows = []
    for e in entries:
        an = e.get("cmeAnalyses") or []
        a = next((x for x in an if x.get("isMostAccurate")), an[0] if an else {})
        enlil = a.get("enlilList") or []
        links = {m[1] for x in (e.get("linkedEvents") or [])
                 if (m := re.search(r"-([A-Z]+)-\d+$", x.get("activityID", "")))}
        lon, half = a.get("longitude"), a.get("halfAngle")
        cone = lon is not None and half is not None and abs(lon) <= half
        hit = any(x.get("isEarthGB") or x.get("estimatedShockArrivalTime") for x in enlil)
        t = pd.to_datetime(e["startTime"], utc=True).tz_localize(None)
        rows.append({"cme_id": e["activityID"], "cme_start_time": t, "cme_date": t.normalize(),
                     "latitude": a.get("latitude"), "longitude": lon, "half_angle": half, "speed": a.get("speed"),
                     "type": a.get("type"), "technique": a.get("measurementTechnique"), "cone_covers_earth": cone,
                     "enlil_earth": hit, "earth_directed": cone or hit, "linked_ips": "IPS" in links,
                     "linked_gst": "GST" in links, "source_location": e.get("sourceLocation") or None,
                     "link": e.get("link")})
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2010-01")
    p.add_argument("--end", default=None, help="last month, YYYY-MM (default: last month)")
    p.add_argument("--refresh", action="store_true", help="download months that are already on disk again")
    p.add_argument("--parse-only", action="store_true")
    args = p.parse_args()
    root = os.environ.get("SOLARIS_ARCHIVE_ROOT")
    if not root:
        raise SystemExit("SOLARIS_ARCHIVE_ROOT is not set (source ~/.solaris_env)")
    raw = Path(root) / "space_weather" / "ccmc" / "donki_cme"
    raw.mkdir(parents=True, exist_ok=True)
    end = pd.Period(args.end, "M") if args.end else pd.Period(pd.Timestamp.today(), "M") - 1

    parts = []
    for m in pd.period_range(pd.Period(args.start, "M"), end, freq="M"):
        f = raw / f"{m}.json"
        if not args.parse_only and (args.refresh or not f.exists()):
            r = requests.get(URL.format(s=m.start_time, e=m.end_time), timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        if f.exists():
            parts.append(catalog(json.loads(f.read_text() or "[]")))
    cat = pd.concat([x for x in parts if len(x)], ignore_index=True).drop_duplicates("cme_id")
    cat = cat.sort_values("cme_start_time").reset_index(drop=True)
    out = default_data_dir() / "events" / "donki_cme.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(out, index=False)
    ed = cat[cat.earth_directed]
    print(f"wrote {out}: {len(cat)} CMEs, {len(ed)} Earth-directed (cone {int(cat.cone_covers_earth.sum())}, "
          f"ENLIL Earth {int(cat.enlil_earth.sum())}); {cat.cme_date.min():%Y-%m-%d} .. {cat.cme_date.max():%Y-%m-%d}")
    print("Earth-directed per year:", ed.groupby(ed.cme_date.dt.year).size().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
