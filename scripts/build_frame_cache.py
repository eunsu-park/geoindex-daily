"""Build the 1024² uint8 frame cache for the lead-1 image arms (resumable, parallel).

Reads `slot_table.parquet`, writes one `.npy` per slot and channel under
`$GEOINDEX_DAILY_DATA/frames1024/<source>/<YYYY>/<YYYYMMDD_HH>.npy`, plus `calibration.json`
and `index.parquet`. Needs the archive (SOLARIS_ARCHIVE_ROOT) — on egghouse-gpu
`~/NAS/archive`, on the laptop `~/Archive`.

    # once, on a host with the database: the previous C2 frame of every slot (running difference)
    python scripts/build_frame_cache.py prev
    # calibration constants from sample frames (training years), then the cache
    python scripts/build_frame_cache.py calibrate --n 40
    python scripts/build_frame_cache.py build --workers 6 [--start 2010-09-01 --end 2025-12-31]
    python scripts/build_frame_cache.py index          # rebuild index.parquet from what is on disk
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.frames import (  # noqa: E402
    SOURCES, cache_path, calibrate, lasco_running_difference, load_calibration, load_frame,
    preprocess_aia, preprocess_hmi, preprocess_lasco, save_calibration,
)
from geoindex_daily.slots import default_archive_root, load_slot_table  # noqa: E402

SLOT_SOURCES = ("aia193", "aia211", "hmi_m45", "lasco_c2")


def slots_frame(data_dir: Path, start=None, end=None) -> pd.DataFrame:
    t = load_slot_table(data_dir / "slot_table.parquet")
    t = t[t.source.isin(SLOT_SOURCES) & t.file_path.notna()]
    if start:
        t = t[t.date >= pd.Timestamp(start)]
    if end:
        t = t[t.date <= pd.Timestamp(end)]
    t = t.assign(slot_time=t.date + pd.to_timedelta(t.slot_hour, unit="h"))
    prev = data_dir / "lasco_c2_prev.parquet"
    if prev.exists():
        pv = pd.read_parquet(prev)[["slot_time", "prev_path"]]
        t = t.merge(pv, on="slot_time", how="left")
    else:
        t["prev_path"] = None
    return t.reset_index(drop=True)


def cmd_prev(args, data_dir):
    """The C2 Orange frame 6–30 min before each slot's frame, from the catalog (needs the DB)."""
    from geoindex_daily.db import connect
    t = load_slot_table(data_dir / "slot_table.parquet")
    t = t[(t.source == "lasco_c2") & t.file_path.notna()]
    with connect("solar_images") as conn:
        cur = conn.cursor()
        cur.execute("select datetime, file_path from lasco where camera='c2' and filter='Orange' order by datetime")
        cat = pd.DataFrame(cur.fetchall(), columns=["datetime", "file_path"])
    cat["datetime"] = pd.to_datetime(cat["datetime"])
    times = cat["datetime"].to_numpy()
    sel = pd.to_datetime(t["datetime"]).to_numpy()
    rows = []
    for st, ft in zip(t.date + pd.to_timedelta(t.slot_hour, unit="h"), sel):
        lo, hi = ft - np.timedelta64(30, "m"), ft - np.timedelta64(6, "m")
        j = np.searchsorted(times, hi, side="right") - 1
        ok = j >= 0 and times[j] >= lo
        rows.append({"slot_time": st, "prev_path": cat.file_path.iloc[j] if ok else None,
                     "prev_datetime": cat.datetime.iloc[j] if ok else pd.NaT})
    out = pd.DataFrame(rows)
    out.to_parquet(data_dir / "lasco_c2_prev.parquet", index=False)
    print(f"wrote lasco_c2_prev.parquet: {len(out)} slots, {out.prev_path.notna().sum()} with a previous frame")
    return 0


def cmd_calibrate(args, data_dir):
    root = default_archive_root()
    t = slots_frame(data_dir, "2011-01-01", "2019-12-31")
    rng = np.random.default_rng(0)
    frames, pairs = {}, []
    for src in ("aia193", "aia211", "lasco_c2"):
        g = t[t.source == src]
        pick = g.iloc[rng.choice(len(g), min(args.n, len(g)), replace=False)]
        frames[src] = []
        for r in pick.itertuples():
            d, h = load_frame(root / r.file_path)
            frames[src].append((d, h))
            if src == "lasco_c2" and r.prev_path:
                q, hq = load_frame(root / r.prev_path)
                if q.shape == d.shape:
                    pairs.append((d, h["EXPTIME"], q, hq["EXPTIME"]))
        print(f"  {src}: {len(frames[src])} frames")
    calib = calibrate(frames, pairs)
    save_calibration(data_dir / "frames1024", calib)
    print("calibration:", calib)
    return 0


_G = {}


def _init(root, cache_dir, calib):
    _G.update(root=Path(root), cache_dir=Path(cache_dir), calib=calib)


def _one(row):
    """Process one slot row → list of (source, slot_time, path, status)."""
    root, cache_dir, calib = _G["root"], _G["cache_dir"], _G["calib"]
    src, st, fp, prev = row
    out = cache_path(cache_dir, src, st)
    res = []
    try:
        todo = [src] + (["lasco_c2_rd"] if src == "lasco_c2" else [])
        outs = {s: cache_path(cache_dir, s, st) for s in todo}
        if all(p.exists() for p in outs.values()):
            return [(s, st, str(p), "exists") for s, p in outs.items()]
        d, h = load_frame(root / fp)
        if src in ("aia193", "aia211"):
            arr = preprocess_aia(d, h["EXPTIME"], calib[src]["hi"])
        elif src == "hmi_m45":
            arr = preprocess_hmi(d, h["CROTA2"] or 0.0, calib["hmi_m45"]["bmax"])
        else:
            arr = preprocess_lasco(d, h["EXPTIME"], calib["lasco_c2"]["hi"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, arr)
        res.append((src, st, str(out), "ok"))
        if src == "lasco_c2":
            o2 = outs["lasco_c2_rd"]
            if prev:
                q, hq = load_frame(root / prev)
                if q.shape == d.shape:
                    o2.parent.mkdir(parents=True, exist_ok=True)
                    np.save(o2, lasco_running_difference(d, h["EXPTIME"], q, hq["EXPTIME"], calib["lasco_c2_rd"]["scale"]))
                    res.append(("lasco_c2_rd", st, str(o2), "ok"))
                else:
                    res.append(("lasco_c2_rd", st, str(o2), "shape-mismatch"))
            else:
                res.append(("lasco_c2_rd", st, str(o2), "no-previous-frame"))
    except Exception as e:  # noqa: BLE001 — keep the pool alive, record the failure
        res.append((src, st, str(out), f"error: {type(e).__name__}: {e}"))
    return res


def cmd_build(args, data_dir):
    root = default_archive_root()
    cache_dir = data_dir / "frames1024"
    calib = load_calibration(cache_dir)
    t = slots_frame(data_dir, args.start, args.end)
    if args.sources:
        t = t[t.source.isin(args.sources)]
    rows = [(a, b, c, (None if pd.isna(d) else d)) for a, b, c, d in zip(t.source, t.slot_time, t.file_path, t.prev_path)]
    print(f"{len(rows)} slot frames, archive {root}, cache {cache_dir}, {args.workers} workers, calib {calib}")
    t0, done, log = time.time(), 0, []
    with Pool(args.workers, initializer=_init, initargs=(root, cache_dir, calib)) as pool:
        for res in pool.imap_unordered(_one, rows, chunksize=4):
            log.extend(res)
            done += 1
            if done % 500 == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                bad = sum(1 for r in log if r[3].startswith("error"))
                print(f"  {done}/{len(rows)} slots, {rate:.1f}/s, eta {(len(rows) - done) / max(rate, 1e-6) / 60:.0f} min, "
                      f"{bad} errors", flush=True)
    idx = pd.DataFrame(log, columns=["source", "slot_time", "path", "status"])
    idx.to_parquet(cache_dir / "build_log.parquet", index=False)
    print(idx.status.str.split(":").str[0].value_counts().to_string())
    return cmd_index(args, data_dir)


def cmd_index(args, data_dir):
    cache_dir = data_dir / "frames1024"
    rows = []
    for src in SOURCES:
        for p in sorted((cache_dir / src).glob("*/*.npy")):
            rows.append({"source": src, "slot_time": pd.to_datetime(p.stem, format="%Y%m%d_%H"),
                         "path": str(p.relative_to(cache_dir))})
    idx = pd.DataFrame(rows)
    idx.to_parquet(cache_dir / "index.parquet", index=False)
    print("index.parquet:", idx.groupby("source").size().to_dict() if len(idx) else "empty")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prev", "calibrate", "build", "index"])
    p.add_argument("--n", type=int, default=40, help="calibrate: frames per channel")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--sources", nargs="*", default=None, help="build: subset of the slot sources")
    args = p.parse_args()
    data_dir = default_data_dir()
    return {"prev": cmd_prev, "calibrate": cmd_calibrate, "build": cmd_build, "index": cmd_index}[args.command](args, data_dir)


if __name__ == "__main__":
    sys.exit(main())
