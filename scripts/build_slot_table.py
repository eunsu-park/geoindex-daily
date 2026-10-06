"""Build the per-day image slot table (nearest frame per day × slot × source) → parquet.

Needs the SOLARIS_DB_* env vars; the NAS mount only with --check-files.

    python scripts/build_slot_table.py                        # → $GEOINDEX_DAILY_DATA/slot_table.parquet
    python scripts/build_slot_table.py --tol-min 180 --check-files
    python scripts/build_slot_table.py --start 2010-09-01 --end 2024-12-31 --out /tmp/slots.parquet
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.db import connect  # noqa: E402
from geoindex_daily.slots import (  # noqa: E402
    SLOT_HOURS_2, SLOT_HOURS_4, SOURCES, TARGET_SOURCES, build_slot_table, check_files,
    day_completeness, save_slot_table, window_counts,
)


def report(table: pd.DataFrame, lengths=range(1, 8)) -> None:
    all_sources = list(SOURCES)
    print("\n== slot fill rate per source (fraction of day × slot cells with a frame)")
    fill = table.assign(ok=table.file_path.notna()).pivot_table(
        index="source", columns="slot_hour", values="ok", aggfunc="mean").reindex(all_sources)
    print(fill.round(3).to_string())
    print("\n== |offset| minutes per source: median / p90 / max over filled cells")
    off = table.dropna(subset=["file_path"]).assign(a=lambda d: d.offset_min.abs()).groupby("source")["a"]
    print(pd.DataFrame({"median": off.median(), "p90": off.quantile(0.9), "max": off.max()})
          .reindex(all_sources).round(1).to_string())
    n_days = table["date"].nunique()
    for label, hours in (("4 slots (00/06/12/18)", SLOT_HOURS_4), ("2 slots (00/12)", SLOT_HOURS_2)):
        comp = day_completeness(table, all_sources, hours)
        comp["target4"] = comp[list(TARGET_SOURCES)].all(axis=1)
        print(f"\n== complete days, {label}, of {n_days}")
        print(comp.drop(columns="all").sum().rename("days").to_frame().T.to_string())
        by_year = comp.groupby(comp.index.year)["target4"].agg(["size", "sum"])
        by_year.columns = ["days", "target4_complete"]
        print(f"\n== target four sources ({', '.join(TARGET_SOURCES)}) complete days by year, {label}")
        print(by_year.T.to_string())
        print(f"\n== issue days with L consecutive complete days (target four sources), {label}")
        print(window_counts(comp["target4"], lengths).T.to_string())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="parquet path (default: data dir/slot_table.parquet)")
    p.add_argument("--start", default="2010-05-13", help="YYYY-MM-DD inclusive (SDO era default)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclusive (default: yesterday)")
    p.add_argument("--tol-min", type=float, default=180.0, help="max |frame − slot| in minutes")
    p.add_argument("--check-files", action="store_true", help="stat every selected frame on the NAS")
    args = p.parse_args()

    end = args.end or (date.today() - timedelta(days=1)).isoformat()
    out = Path(args.out) if args.out else default_data_dir() / "slot_table.parquet"
    with connect("solar_images") as conn:
        table = build_slot_table(conn, args.start, end, SLOT_HOURS_4, args.tol_min)
    if args.check_files:
        table["exists"] = check_files(table)
        missing = table[table.file_path.notna() & ~table.exists]
        print(f"files checked: {int(table.file_path.notna().sum())} selected, {len(missing)} missing on disk")
        if len(missing):
            print(missing.groupby("source").size().rename("missing").to_string())
            table.loc[missing.index, ["datetime", "file_path", "offset_min"]] = [pd.NaT, None, float("nan")]
    path = save_slot_table(table, out)
    print(f"wrote {path}: {len(table)} rows, {table.date.nunique()} days {args.start} .. {end}, "
          f"tol ±{args.tol_min:g} min")
    report(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
