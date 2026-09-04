"""Pull the SWPC PRF 27-day outlook table (loaded by solaris-data) into the local parquet.

`eval_swpc.py` and `compare_on_prf_issues.py` read `<data dir>/swpc/prf_outlook.parquet`;
this refreshes it from `space_weather.swpc_prf_outlook`. Needs the DB, not the NAS.

    python scripts/load_swpc_prf.py
    python scripts/load_swpc_prf.py --start 2010-01-01 --end 2025-12-31
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.db import connect  # noqa: E402
from geoindex_daily.swpc import load_prf_outlook  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help="issue-date lower bound YYYY-MM-DD")
    p.add_argument("--end", default=None, help="issue-date upper bound YYYY-MM-DD")
    p.add_argument("--out", default=None, help="parquet path (default data dir/swpc/prf_outlook.parquet)")
    args = p.parse_args()

    out = Path(args.out) if args.out else default_data_dir() / "swpc" / "prf_outlook.parquet"
    with connect("space_weather") as conn:
        df = load_prf_outlook(conn, args.start, args.end)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} rows, {df.issue_date.nunique()} issues "
          f"{df.issue_date.min().date()} .. {df.issue_date.max().date()} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
