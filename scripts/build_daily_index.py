"""Build the daily index table (Ap, Kp, F10.7, SN, Dst) from OMNI hourly → parquet.

Needs the SOLARIS_DB_* env vars and network access to the DB; no NAS mount.

    python scripts/build_daily_index.py                      # → $GEOINDEX_DAILY_DATA/daily_index.parquet
    python scripts/build_daily_index.py --out /tmp/d.parquet --start 2010-01-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import build, load_daily  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="parquet path (default: data dir/daily_index.parquet)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", default=None, help="YYYY-MM-DD exclusive")
    args = p.parse_args()

    path = build(args.out, args.start, args.end)
    d = load_daily(path)
    short = (d.n_ap_hours < 24).sum()
    print(f"wrote {path}: {len(d)} days {d.index.min().date()} .. {d.index.max().date()}, "
          f"{short} day(s) with <24 valid ap hours")
    print(d[["ap", "kp", "f107", "sn", "dst"]].describe().round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
