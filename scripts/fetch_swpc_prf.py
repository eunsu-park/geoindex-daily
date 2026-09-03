"""Mirror the weekly PRF PDFs from NCEI and parse their 27-day Ap/F10.7 outlooks.

    python scripts/fetch_swpc_prf.py --start 2020-01 --end 2020-03      # sample
    python scripts/fetch_swpc_prf.py --start 2010-01 --end 2025-12      # full (~830 PDFs, ~250 MB)

PDFs go to `<data dir>/swpc/prf/YYYY/prfNNNN.pdf` (skipped if present); the parsed long
table to `<data dir>/swpc/prf_outlook.parquet` (issue_date, target_date, lead, f107, ap, kp).
Parse failures are listed at the end and do not stop the run. Needs internet, not the NAS.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.swpc import URL_PRF_DIR, parse_prf_outlook, pdf_text, prf_index  # noqa: E402


def months(start: str, end: str):
    cur, last = pd.Period(start, "M"), pd.Period(end, "M")
    while cur <= last:
        yield cur.year, cur.month
        cur += 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end", required=True, help="YYYY-MM")
    p.add_argument("--out", default=None, help="parquet path (default data dir/swpc/prf_outlook.parquet)")
    p.add_argument("--no-parse", action="store_true", help="download only")
    args = p.parse_args()

    root = default_data_dir() / "swpc"
    out = Path(args.out) if args.out else root / "prf_outlook.parquet"
    sess = requests.Session()
    fetch = lambda url: sess.get(url, timeout=60).text  # noqa: E731

    tables, failures, n_dl = [], [], 0
    for y, m in months(args.start, args.end):
        try:
            names = prf_index(y, m, fetch)
        except requests.RequestException as e:
            failures.append((f"{y}-{m:02d}", f"index: {e}")); continue
        for name in names:
            pdf = root / "prf" / f"{y}" / name
            if not pdf.exists():
                pdf.parent.mkdir(parents=True, exist_ok=True)
                r = sess.get(URL_PRF_DIR.format(y=y, m=m) + name, timeout=120)
                if r.status_code != 200:
                    failures.append((name, f"http {r.status_code}")); continue
                pdf.write_bytes(r.content); n_dl += 1
            if args.no_parse:
                continue
            try:
                tables.append(parse_prf_outlook(pdf_text(pdf)))
            except Exception as e:  # noqa: BLE001 — keep going, report at the end
                failures.append((name, str(e)[:120]))
        print(f"{y}-{m:02d}: {len(names)} PRFs ({n_dl} downloaded so far, {len(failures)} failures)")

    if tables:
        df = pd.concat(tables, ignore_index=True).sort_values(["issue_date", "target_date"])
        if out.exists():
            old = pd.read_parquet(out)
            df = (pd.concat([old, df]).drop_duplicates(["issue_date", "target_date"], keep="last")
                  .sort_values(["issue_date", "target_date"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"outlook rows: {len(df)}  issues {df.issue_date.min().date()} .. {df.issue_date.max().date()} → {out}")
    for name, why in failures:
        print(f"  FAIL {name}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
