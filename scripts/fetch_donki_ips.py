"""Collect the NASA DONKI interplanetary shocks observed at Earth (IPS, location Earth) and their linked events.

Used by the physical event analysis (work order 2026-11-17, inbox/20261118): the observed shock arrival time and the
CMEs DONKI links to it. Source: https://ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS (no key, per calendar month). Raw JSON
under $SOLARIS_ARCHIVE_ROOT/space_weather/ccmc/donki_ips/<YYYY-MM>.json; table at
$GEOINDEX_DAILY_DATA/events/donki_ips_earth.parquet (one row per shock: time, linked CMEs / HSS / storms, instruments).

    python scripts/fetch_donki_ips.py --start 2022-01 --end 2025-12
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

URL = "https://ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS?startDate={s:%Y-%m-%d}&endDate={e:%Y-%m-%d}&location=Earth&catalog=ALL"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2022-01")
    p.add_argument("--end", default="2025-12")
    p.add_argument("--parse-only", action="store_true")
    args = p.parse_args()
    root = os.environ.get("SOLARIS_ARCHIVE_ROOT")
    if not root:
        raise SystemExit("SOLARIS_ARCHIVE_ROOT is not set (source ~/.solaris_env)")
    raw = Path(root) / "space_weather" / "ccmc" / "donki_ips"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in pd.period_range(args.start, args.end, freq="M"):
        f = raw / f"{m}.json"
        if not args.parse_only and not f.exists():
            r = requests.get(URL.format(s=m.start_time, e=m.end_time), timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        for e in json.loads(f.read_text() or "[]") if f.exists() else []:
            links = [x["activityID"] for x in (e.get("linkedEvents") or [])]
            kind = lambda k: [a for a in links if re.search(rf"-{k}-\d+$", a)]  # noqa: E731
            rows.append({"ips_id": e["activityID"], "shock_time": pd.to_datetime(e["eventTime"], utc=True).tz_localize(None),
                         "linked_cmes": ";".join(kind("CME")), "linked_hss": ";".join(kind("HSS")),
                         "linked_gst": ";".join(kind("GST")),
                         "instruments": ";".join(i["displayName"] for i in e.get("instruments") or []), "link": e.get("link")})
    df = pd.DataFrame(rows).drop_duplicates("ips_id").sort_values("shock_time")
    out = default_data_dir() / "events" / "donki_ips_earth.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} shocks at Earth, {int((df.linked_cmes != '').sum())} linked to a CME, "
          f"{int((df.linked_hss != '').sum())} to an HSS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
