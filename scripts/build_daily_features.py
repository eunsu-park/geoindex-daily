"""Build the daily feature table for the lead-1 track → parquet.

Needs daily_index.parquet (scripts/build_daily_index.py) and the SOLARIS_DB_* env vars for
the `space_weather` database; no NAS.

    python scripts/build_daily_features.py                 # → $GEOINDEX_DAILY_DATA/daily_features.parquet
    python scripts/build_daily_features.py --out /tmp/f.parquet --start 2010-01-01 --end 2026-01-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.db import connect  # noqa: E402
from geoindex_daily.features import SW_COLUMNS, XRAY_COLUMNS, build_daily_features  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--daily-index", default=None, help="default: data dir/daily_index.parquet")
    p.add_argument("--out", default=None, help="default: data dir/daily_features.parquet")
    p.add_argument("--start", default=None, help="YYYY-MM-DD inclusive (default: the daily index's range)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD exclusive")
    args = p.parse_args()

    data = default_data_dir()
    daily_index = Path(args.daily_index) if args.daily_index else data / "daily_index.parquet"
    out = Path(args.out) if args.out else data / "daily_features.parquet"
    with connect("space_weather") as conn:
        df = build_daily_features(daily_index, conn, out, args.start, args.end)
    print(f"wrote {out}: {len(df)} days {df.index.min().date()} .. {df.index.max().date()}")
    cols = ["ap", "f107", "sn"] + XRAY_COLUMNS[:-1] + SW_COLUMNS[:-1]
    miss = df[cols].isna().sum()
    print("missing days per column:\n" + miss.to_string())
    print(f"days with < 1000 good X-ray minutes: {(df.n_xray_min < 1000).sum()}; "
          f"with < 12 valid solar-wind hours: {(df.n_sw_hours < 12).sum()}")
    print(df[cols].describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
