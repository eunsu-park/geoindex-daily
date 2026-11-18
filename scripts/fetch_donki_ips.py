"""Collect the NASA DONKI interplanetary shocks observed at Earth (IPS, location Earth) and the high-speed streams seen
near Earth (HSS observed by L1 spacecraft: ACE, DSCOVR, Wind, SOHO), with their linked events.

Used by the physical event analysis (work order 2026-11-17, inbox/20261118): the observed shock arrival time and the
CMEs DONKI links to it. Source: https://ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS (no key, per calendar month). Raw JSON
under $SOLARIS_ARCHIVE_ROOT/space_weather/ccmc/donki_ips/<YYYY-MM>.json; table at
$GEOINDEX_DAILY_DATA/events/donki_ips_earth.parquet (one row per shock: time, linked CMEs / HSS / storms, instruments);
HSS raw JSON under …/ccmc/donki_hss/, table events/donki_hss_earth.parquet (HSS seen only by STEREO is left out).

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
URL_HSS = "https://ccmc.gsfc.nasa.gov/DONKI/WS/get/HSS?startDate={s:%Y-%m-%d}&endDate={e:%Y-%m-%d}"
L1 = ("ACE", "DSCOVR", "WIND", "SOHO")


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
    raw_h = Path(root) / "space_weather" / "ccmc" / "donki_hss"
    raw_h.mkdir(parents=True, exist_ok=True)
    hs = []
    for m in pd.period_range(args.start, args.end, freq="M"):
        f = raw_h / f"{m}.json"
        if not args.parse_only and not f.exists():
            r = requests.get(URL_HSS.format(s=m.start_time, e=m.end_time), timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        for e in json.loads(f.read_text() or "[]") if f.exists() else []:
            inst = [i["displayName"] for i in e.get("instruments") or []]
            if not any(k in x.upper() for x in inst for k in L1):
                continue
            hs.append({"hss_id": e["hssID"], "hss_time": pd.to_datetime(e["eventTime"], utc=True).tz_localize(None),
                       "linked": ";".join(x["activityID"] for x in (e.get("linkedEvents") or [])), "instruments": ";".join(inst)})
    hd = pd.DataFrame(hs).drop_duplicates("hss_id").sort_values("hss_time")
    hd.to_parquet(default_data_dir() / "events" / "donki_hss_earth.parquet", index=False)
    print(f"wrote donki_hss_earth.parquet: {len(hd)} high-speed streams seen at L1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
