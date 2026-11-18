"""Collect NOAA SWPC's 3-Day Geomagnetic Forecast (historical issues) from the NCEI archive and parse it.

Work order 2026-11-17 (inbox/20261118, "NOAA SWPC 3-hourly Kp → ap Benchmark"): the 3-hourly Kp forecast of the
3-Day Geomagnetic Forecast (`geomag_forecast/`, `YYYYMMDDgeomag_forecast.txt`), converted to ap with the official
28-step Kp–ap table, is the benchmark for the l6 3-hourly ap models. Only this product is used (not the separate
`3day_forecast/`). Source: https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/geomag_forecast/
(monthly folders; the archive has gaps). Raw files are kept unchanged under
$SOLARIS_ARCHIVE_ROOT/space_weather/swpc/geomag_forecast/YYYY/MM/ with a manifest (URL, SHA-256).

Parsed per issue: the `:Issued:` time, the three target dates, 3 × 8 Kp values (00-03UT … 21-00UT), the official
`Predicted Ap` for the three days; the Kp is converted to ap (28-step table, tolerance 0.02).

    python scripts/fetch_swpc_geomag_forecast.py --out ~/Projects/GeoIndex/daily/benchmark_swpc_ap3h
    python scripts/fetch_swpc_geomag_forecast.py --parse-only --out …
"""
import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/geomag_forecast/"
# NOAA / NCEI Kp–ap table (https://www.ngdc.noaa.gov/geomag/indices/kp_ap.html): Kp in thirds → ap
KP_AP = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 9, 8: 12, 9: 15, 10: 18, 11: 22, 12: 27, 13: 32, 14: 39,
         15: 48, 16: 56, 17: 67, 18: 80, 19: 94, 20: 111, 21: 132, 22: 154, 23: 179, 24: 207, 25: 236, 26: 300, 27: 400}
BINS = ["00-03UT", "03-06UT", "06-09UT", "09-12UT", "12-15UT", "15-18UT", "18-21UT", "21-00UT"]


def kp_to_ap(kp: float, tol: float = 0.02):
    """ap of the nearest of the 28 Kp steps (thirds), or None if the value is off-grid by more than tol."""
    k = round(kp * 3)
    if abs(kp - k / 3) > tol or k not in KP_AP:
        return None
    return KP_AP[k]


def parse(text: str):
    """→ (issue time, [3 target dates], 3×8 Kp array, [3 predicted Ap]) or raises ValueError."""
    m = re.search(r":Issued:\s+(\d{4} \w{3} \d{2} \d{4}) UTC", text)
    if not m:
        raise ValueError("no :Issued:")
    issued = pd.to_datetime(m[1], format="%Y %b %d %H%M")
    m = re.search(r"Predicted Ap\s+\d{2} \w{3}-\d{2} \w{3}\s+(\d+)-(\d+)-(\d+)", text)
    ap_pred = [int(x) for x in m.groups()] if m else [np.nan] * 3
    lines = text.splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.startswith("NOAA Kp index forecast"))
    head = lines[i + 1].split()                                      # e.g. Mar 02 Mar 03 Mar 04
    if len(head) != 6:
        raise ValueError(f"Kp header: {lines[i + 1]!r}")
    dates = []
    for mon, day in zip(head[0::2], head[1::2]):
        d = pd.to_datetime(f"{issued.year} {mon} {day}", format="%Y %b %d")
        if d < issued.normalize() - pd.Timedelta(days=1):           # target in January, issued in December
            d = d + pd.DateOffset(years=1)
        dates.append(d)
    kp = np.full((3, 8), np.nan)
    for b, name in enumerate(BINS):
        row = next((ln for ln in lines[i + 2: i + 12] if ln.strip().startswith(name)), None)
        if row is None:
            raise ValueError(f"missing row {name}")
        vals = row.split()[1:]
        if len(vals) != 3:
            raise ValueError(f"row {name}: {row!r}")
        kp[:, b] = [float(re.sub(r"\(.*\)", "", v)) for v in vals]   # values may carry a (G1)-style note
    return issued, dates, kp, ap_pred


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="2022-01")
    p.add_argument("--end", default="2026-01")
    p.add_argument("--parse-only", action="store_true")
    args = p.parse_args()
    root = os.environ.get("SOLARIS_ARCHIVE_ROOT")
    if not root:
        raise SystemExit("SOLARIS_ARCHIVE_ROOT is not set (source ~/.solaris_env)")
    raw = Path(root) / "space_weather" / "swpc" / "geomag_forecast"
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    (args.out / "processed").mkdir(parents=True, exist_ok=True)
    (args.out / "qc").mkdir(parents=True, exist_ok=True)

    cov = []
    if not args.parse_only:
        for m in pd.period_range(args.start, args.end, freq="M"):
            url = f"{BASE}{m.year}/{m.month:02d}/"
            r = requests.get(url, timeout=120)
            names = sorted(set(re.findall(r'href="(\d{8}geomag_forecast\.txt)"', r.text))) if r.ok else []
            cov.append({"month": str(m), "folder_status": r.status_code, "n_files_listed": len(names)})
            d = raw / f"{m.year}" / f"{m.month:02d}"
            for n in names:
                f = d / n
                if f.exists():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                for _ in range(3):
                    rr = requests.get(url + n, timeout=120)
                    if rr.ok:
                        f.write_bytes(rr.content); break
                    time.sleep(3)
        pd.DataFrame(cov).to_csv(args.out / "manifests" / "noaa_archive_months.csv", index=False)

    rows, files, fails, anomalies = [], [], [], []
    for f in sorted(raw.rglob("*geomag_forecast.txt")):
        y, mo = f.parent.parent.name, f.parent.name
        url = f"{BASE}{y}/{mo}/{f.name}"
        b = f.read_bytes()
        files.append({"source_file": f.name, "source_url": url, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)})
        if f.name[:6] != f"{y}{mo}":                                 # NCEI misfiling (e.g. a 2025 file in 2024/11/)
            twin = raw / f.name[:4] / f.name[4:6] / f.name
            same = twin.exists() and twin.read_bytes() == b
            anomalies.append({"source_file": f.name, "source_url": url, "folder": f"{y}/{mo}",
                              "note": "copy of the file in its own month folder; not used" if same else "only copy; used"})
            if same:
                continue
        try:
            issued, dates, kp, ap_pred = parse(b.decode("latin-1"))
        except (ValueError, StopIteration) as e:
            fails.append({"source_file": f.name, "source_url": url, "error": str(e)}); continue
        issue_date = pd.Timestamp(f.name[:8])
        for k in range(3):
            for s in range(8):
                ap = kp_to_ap(kp[k, s])
                start = dates[k] + pd.Timedelta(hours=3 * s)
                rows.append({"issue_date": issue_date.date(), "swpc_issue_utc": issued, "source_file": f.name,
                             "source_url": url, "target_day_k": k + 1, "step_in_day": s + 1, "lead_step_3h": 8 * k + s + 1,
                             "target_start_utc": start, "target_end_utc": start + pd.Timedelta(hours=3),
                             "forecast_kp": kp[k, s], "swpc_ap_converted": ap, "swpc_daily_ap_official": ap_pred[k],
                             "parse_status": "ok" if ap is not None else "kp_off_grid"})
    df = pd.DataFrame(rows)
    # QC: issue time on the file's date, before 00 UT of the first target day; targets = issue date + 1, 2, 3
    df["qc_issue_before_target"] = df.swpc_issue_utc < pd.to_datetime(df.issue_date) + pd.Timedelta(days=1)
    df["qc_target_dates"] = (df.target_start_utc.dt.normalize() - pd.to_datetime(df.issue_date)).dt.days == df.target_day_k
    df.to_csv(args.out / "processed" / "swpc_3hourly_kp_to_ap.csv", index=False)
    pd.DataFrame(files).to_csv(args.out / "manifests" / "noaa_raw_files.csv", index=False)
    pd.DataFrame(fails, columns=["source_file", "source_url", "error"]).to_csv(args.out / "qc" / "parser_failures.csv", index=False)
    pd.DataFrame(anomalies, columns=["source_file", "source_url", "folder", "note"]).to_csv(
        args.out / "qc" / "archive_anomalies.csv", index=False)
    per = df.groupby("issue_date").agg(n=("swpc_ap_converted", "count"), ok=("parse_status", lambda s: (s == "ok").all()),
                                       qi=("qc_issue_before_target", "all"), qt=("qc_target_dates", "all"))
    print(f"files {len(files)}, parsed issues {len(per)}, parser failures {len(fails)}, issues with 24 valid ap "
          f"{int(((per.n == 24) & per.ok).sum())}, issue-time QC fail {int((~per.qi).sum())}, target-date QC fail "
          f"{int((~per.qt).sum())}, off-grid Kp {int((df.parse_status == 'kp_off_grid').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
