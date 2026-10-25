"""Fill the missing LASCO C2 running-difference slots so the W41 issue days keep all 12 slots a day.

Co-authors' answer to Q4 (2026-10-25): LASCO enters as the running difference, 12 frames a day; agreed
option (b): keep the W41 issue days (1,273 split-A test days) by filling the slots that have no
running-difference frame instead of dropping the day. Rule, per day of the period and slot hour:
  1. the slot's own cached frame (lasco_c2_rd12) when it exists;
  2. else the cached frame of the nearest slot in time within MAX_H hours (ties: the earlier one);
  3. else a "no change" frame: the uint8 value of a zero difference, 127, everywhere.
The filled channel lasco_c2_rd12f is a folder of symlinks (rule 1 and 2) and one neutral file (rule 3);
slot_table_lasco12f.parquet / lasco_c2_prev12f.parquet record, per slot, the source slot and its offset,
so every filled slot can be counted per model and day.

    python scripts/fill_lasco12.py [--start 2010-09-01 --end 2025-12-31]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.frames import cache_path  # noqa: E402

HOURS = list(range(0, 24, 2))
MAX_H = 24
SRC, DST = "lasco_c2_rd12", "lasco_c2_rd12f"
NEUTRAL = 127          # to_uint8(0, -scale, scale): a zero running difference


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2010-09-01")
    p.add_argument("--end", default="2025-12-31")
    args = p.parse_args()
    data = default_data_dir()
    cache = data / "frames1024"
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in pd.date_range(args.start, args.end) for h in HOURS])
    have = np.array([cache_path(cache, SRC, t).exists() for t in times])
    have_t = times[have]
    neutral = cache / DST / "neutral.npy"
    neutral.parent.mkdir(parents=True, exist_ok=True)
    np.save(neutral, np.full((1024, 1024), NEUTRAL, np.uint8))

    rows = []
    for t, ok in zip(times, have):
        if ok:
            src, off = t, 0.0
        else:
            i = have_t.searchsorted(t)
            cands = [have_t[j] for j in (i - 1, i) if 0 <= j < len(have_t)]
            best = min(cands, key=lambda c: (abs((c - t).total_seconds()), c)) if cands else None
            if best is not None and abs((best - t).total_seconds()) <= MAX_H * 3600:
                src, off = best, (best - t).total_seconds() / 3600
            else:
                src, off = None, np.nan
        link = cache_path(cache, DST, t)
        link.parent.mkdir(parents=True, exist_ok=True)
        target = cache_path(cache, SRC, src) if src is not None else neutral
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(target, link.parent), link)
        rows.append({"date": t.normalize(), "slot_hour": t.hour, "slot_time": t,
                     "fill": "own" if ok else ("nearest" if src is not None else "neutral"),
                     "source_slot": src, "offset_h": off})
    tab = pd.DataFrame(rows)
    # the table layout lead1_image.py reads (file_path set for every slot; the prev table says how)
    slots = tab.assign(source="lasco_c2", datetime=tab.source_slot, file_path=DST, offset_min=tab.offset_h * 60)
    slots[["date", "slot_hour", "source", "datetime", "file_path", "offset_min", "fill"]].to_parquet(
        data / "slot_table_lasco12f.parquet", index=False)
    tab.assign(prev_path=tab.fill)[["slot_time", "prev_path", "fill", "source_slot", "offset_h"]].to_parquet(
        data / "lasco_c2_prev12f.parquet", index=False)
    c = tab.fill.value_counts()
    print(f"{len(tab)} slots {args.start} .. {args.end}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    near = tab[tab.fill == "nearest"].offset_h.abs()
    if len(near):
        print(f"nearest-fill offsets (h): median {near.median():.0f}, 90% {near.quantile(.9):.0f}, max {near.max():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
