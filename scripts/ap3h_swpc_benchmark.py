"""NOAA SWPC 3-hourly Kp → ap benchmark for the l6 3-hourly ap models (24 / 48 / 72 h), work order 2026-11-17.

Inputs: the l6 forecasts (npz per horizon, model group, L and seed; read only), the parsed SWPC 3-Day Geomagnetic
Forecast (processed/swpc_3hourly_kp_to_ap.csv from fetch_swpc_geomag_forecast.py), RSGA day-k Ap, the DONKI CME mask.
For every split-A issue date D (forecast origin D+1 00 UT) the model's bins of days D+1 … D+k are matched to the SWPC
forecast issued on D (one issue for all three days) by issue date and 3-hour start time; the observed ap is the
models' own ground truth. Outputs under --out (README.md in the folder describes them):

* manifests/: split-A issue dates, run registry (npz and checkpoint per run), archive coverage by month;
* processed/: model predictions (long, parquet), matched table (parquet; every group, L and seed), daily-Ap cross-check;
* metrics/: 3-hourly overall / by target day / by step / storm and CME bins, derived daily Ap, paired bootstrap;
* qc/: missing issue dates, alignment audit, input-availability audit.

Forecasts: `ens` = the mean of the three seeds' point forecasts expm1(mu) (the forecast reported in W45 / W46);
`seed` rows = each seed's own forecast; `seedmean` = the mean of the three seeds' metrics. ME = predicted − observed.
Paired bootstrap: monthly blocks of issue dates, 2,000 resamples, ΔMAE / ΔCC = GeoIndex − SWPC (negative ΔMAE = GeoIndex
better).

    python scripts/ap3h_swpc_benchmark.py --out ~/Projects/GeoIndex/daily/benchmark_swpc_ap3h
"""
import argparse
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.ap3h import BINS, cme_days  # noqa: E402
from geoindex_daily.daily_index import default_data_dir  # noqa: E402

STORM_AP = 48.0
# horizon (h) → {model group: (folders, tag, model, subset)}; folders relative to $GEOINDEX_DAILY_DATA
RUNS = {24: {"full": (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], "l6", "full", "all"),
             "ts": (["ap3h"], "l6", "ts", "all"),
             "cme": (["ap3h-l6/gpu2", "ap3h-l6/gpu3"], "l6cme", "full", "cme")},
        48: {"full": (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h2", "full", "all"),
             "ts": (["ap3h-h"], "l6h2", "ts", "all"),
             "cme": (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6cmeh2", "full", "cme")},
        72: {"full": (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6h3", "full", "all"),
             "ts": (["ap3h-h"], "l6h3", "ts", "all"),
             "cme": (["ap3h-l6h/gpu2", "ap3h-l6h/gpu3"], "l6cmeh3", "full", "cme")}}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(y, p):
    """MAE, RMSE, CC, ME over the flattened arrays."""
    y, p = np.ravel(y), np.ravel(p)
    return {"n_bins": y.size, "mae": np.abs(p - y).mean(), "rmse": np.sqrt(((p - y) ** 2).mean()),
            "cc": np.corrcoef(p, y)[0, 1] if y.size > 2 and y.std() > 0 and p.std() > 0 else np.nan,
            "me": (p - y).mean()}


def boot(y, a, b, dates, n=2000, seed=0):
    """Monthly-block bootstrap of MAE(a) − MAE(b) and CC(a) − CC(b); y, a, b (N, B) with N issue dates."""
    rng = np.random.default_rng(seed)
    months = pd.PeriodIndex(dates, freq="M")
    blocks = [np.flatnonzero(months == m) for m in months.unique()]
    ea, eb = np.abs(a - y).mean(1), np.abs(b - y).mean(1)
    dm, dc = [], []
    for _ in range(n):
        i = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        dm.append(ea[i].mean() - eb[i].mean())
        yy, aa, bb = y[i].ravel(), a[i].ravel(), b[i].ravel()
        dc.append(np.corrcoef(aa, yy)[0, 1] - np.corrcoef(bb, yy)[0, 1])
    return {"d_mae": ea.mean() - eb.mean(), "d_mae_lo": np.percentile(dm, 2.5), "d_mae_hi": np.percentile(dm, 97.5),
            "d_cc": np.corrcoef(a.ravel(), y.ravel())[0, 1] - np.corrcoef(b.ravel(), y.ravel())[0, 1],
            "d_cc_lo": np.percentile(dc, 2.5), "d_cc_hi": np.percentile(dc, 97.5)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--boot", type=int, default=2000)
    args = p.parse_args()
    data = default_data_dir()
    out = args.out
    for d in ("manifests", "processed", "metrics", "qc"):
        (out / d).mkdir(parents=True, exist_ok=True)

    # ---- SWPC: one issue per date, (issue date, lead step) → converted ap ----------------------------------------
    sw = pd.read_csv(out / "processed" / "swpc_3hourly_kp_to_ap.csv", parse_dates=["issue_date", "swpc_issue_utc",
                                                                                   "target_start_utc"])
    ok_issue = sw.groupby("issue_date").apply(lambda g: len(g) == 24 and (g.parse_status == "ok").all()
                                              and g.qc_issue_before_target.all() and g.qc_target_dates.all())
    sw = sw[sw.issue_date.isin(ok_issue[ok_issue].index)]
    sw_ap = sw.pivot(index="issue_date", columns="lead_step_3h", values="swpc_ap_converted").sort_index()
    sw_kp = sw.pivot(index="issue_date", columns="lead_step_3h", values="forecast_kp").sort_index()
    sw_iss = sw.groupby("issue_date").swpc_issue_utc.first()
    sw_off = sw.pivot_table(index="issue_date", columns="target_day_k", values="swpc_daily_ap_official", aggfunc="first")
    sw_file = sw.groupby("issue_date").source_file.first()

    # ---- model forecasts ----------------------------------------------------------------------------------------
    f3 = pd.read_parquet(data / "features_3h.parquet")["ap"]
    rsga = pd.read_parquet(data / "swpc" / "rsga_ap.parquet")
    reg, longs, fc = [], [], {}
    for h, groups in RUNS.items():
        for grp, (dirs, tag, model, subset) in groups.items():
            for L in range(1, 8):
                files = sorted(f for d in dirs for f in glob.glob(str(data / d / f"{tag}_preds_A_{model}_{subset}_L{L}_s*.npz")))
                assert len(files) == 3, (h, grp, L, files)
                z = [np.load(f) for f in files]
                dates = pd.DatetimeIndex(z[0]["dates"]).normalize()
                y = np.round(z[0]["y"], 6)
                seeds = np.stack([np.expm1(x["mu"]) for x in z])            # (3, N, H)
                fc[(h, grp, L)] = (dates, y, seeds)
                for s, f in enumerate(files):
                    ck = Path(f).parent / "ckpt" / Path(f).name.replace("_preds_", "_").replace(".npz", ".pt")
                    reg.append({"model_horizon_h": h, "model_group": grp, "tag": tag, "input_length_days": L, "seed": s,
                                "predictions_file": str(Path(f).relative_to(data)), "predictions_sha256": sha(Path(f)),
                                "checkpoint_file": str(ck.relative_to(data)) if ck.exists() else "",
                                "n_issue_dates": len(dates), "n_steps": y.shape[1]})
                    H = y.shape[1]
                    longs.append(pd.DataFrame({
                        "issue_date": np.repeat(dates, H), "model_horizon_h": h, "model_group": grp,
                        "input_length_days": L, "seed": s, "lead_step_3h": np.tile(np.arange(1, H + 1), len(dates)),
                        "geoi_ap_pred": seeds[s].ravel(), "observed_ap": y.ravel(),
                        "geoi_sigma_log1p": z[s]["sigma"].ravel(), "run_id": f"{tag}_{grp}_L{L}_s{s}",
                        "checkpoint_id": Path(reg[-1]["checkpoint_file"]).name}))
    lg = pd.concat(longs, ignore_index=True)
    lg["geoi_forecast_origin_utc"] = lg.issue_date + pd.Timedelta(days=1)
    lg["target_day_k"] = (lg.lead_step_3h - 1) // BINS + 1
    lg["target_start_utc"] = lg.geoi_forecast_origin_utc + pd.to_timedelta(3 * (lg.lead_step_3h - 1), unit="h")
    lg.to_parquet(out / "processed" / "geoi_ap3h_predictions_long.parquet", index=False)
    pd.DataFrame(reg).to_csv(out / "manifests" / "run_registry_24_48_72.csv", index=False)

    # ---- manifests and QC --------------------------------------------------------------------------------------
    issue = fc[(24, "full", 1)][0]
    cme = cme_days(issue, data / "events" / "donki_cme.parquet")
    man = pd.DataFrame({"issue_date": issue.date, "forecast_origin_utc": issue + pd.Timedelta(days=1), "split": "A",
                        "observed_ap_available": True, "is_cme_issue_day": cme,
                        "swpc_issue_available": issue.isin(sw_ap.index)})
    man.to_csv(out / "manifests" / "splitA_issue_dates.csv", index=False)
    man[~man.swpc_issue_available].assign(reason="no SWPC geomag_forecast issue in the NCEI archive").to_csv(
        out / "qc" / "missing_issue_dates.csv", index=False)
    mon = man.assign(month=pd.PeriodIndex(issue, freq="M").astype(str)).groupby("month").agg(
        required_issue_days=("issue_date", "size"), swpc_issue_days=("swpc_issue_available", "sum"))
    mon["matched_issue_days"] = mon.swpc_issue_days
    mon.to_csv(out / "manifests" / "noaa_archive_coverage.csv")
    matched = issue[issue.isin(sw_ap.index)]
    # alignment audit: the model's targets vs the 3-hour table, SWPC target dates vs issue date
    al = []
    for h in (24, 48, 72):
        d, y, _ = fc[(h, "full", 1)]
        pos = f3.index.get_indexer(d + pd.Timedelta(days=1))
        obs = np.stack([f3.to_numpy()[i: i + y.shape[1]] for i in pos])
        al.append({"model_horizon_h": h, "n_issue_dates": len(d), "max_abs_obs_diff_vs_3h_table": float(np.nanmax(np.abs(obs - y))),
                   "swpc_target_dates_ok": bool(sw.qc_target_dates.all()), "swpc_issue_before_origin": bool(sw.qc_issue_before_target.all()),
                   "issues_one_swpc_issue_for_all_days": True})
    pd.DataFrame(al).to_csv(out / "qc" / "alignment_audit.csv", index=False)
    ia = pd.DataFrame({"issue_date": matched.date, "swpc_issue_utc": sw_iss.reindex(matched).to_numpy(),
                       "geoi_forecast_origin_utc": matched + pd.Timedelta(days=1)})
    ia["geoi_latest_input_utc"] = ia.geoi_forecast_origin_utc
    ia["geoi_input_lead_over_swpc_h"] = (ia.geoi_latest_input_utc - ia.swpc_issue_utc).dt.total_seconds() / 3600
    ia["geoi_ap_inputs"] = "GFZ definitive / preliminary 3-hour ap (retrospective; not the real-time nowcast)"
    ia["geoi_image_inputs"] = "SDO slots 00/06/12/18 UT (±3 h), LASCO RD slots 00–22 UT (±60 min) of days D−L+1..D"
    ia.to_csv(out / "qc" / "input_availability_audit.csv", index=False)

    # ---- matched table and metrics --------------------------------------------------------------------------------
    mt, rows_o, rows_d, rows_s, rows_x, rows_b, rows_day = [], [], [], [], [], [], []
    for (h, grp, L), (d, y, seeds) in fc.items():
        keep = d.isin(sw_ap.index)
        d, y, seeds = d[keep], y[keep], seeds[:, keep]
        H = y.shape[1]
        sa = sw_ap.loc[d].to_numpy()[:, :H]
        kp = sw_kp.loc[d].to_numpy()[:, :H]
        is_cme = cme_days(d, data / "events" / "donki_cme.parquet")
        ens = seeds.mean(0)
        pers = np.tile(np.stack([f3.to_numpy()[i - BINS: i] for i in f3.index.get_indexer(d + pd.Timedelta(days=1))]), (1, H // BINS))
        base = {"model_horizon_h": h, "model_group": grp, "input_length_days": L, "n_issue_dates": len(d)}
        steps = np.tile(np.arange(1, H + 1), len(d))
        for s in range(3):
            mt.append(pd.DataFrame({
                    "issue_date": np.repeat(d, H), "model_horizon_h": h, "model_group": grp, "input_length_days": L, "seed": s,
                    "swpc_issue_utc": np.repeat(sw_iss.loc[d].to_numpy(), H), "lead_step_3h": steps,
                    "target_day_k": (steps - 1) // BINS + 1,
                    "target_start_utc": np.repeat(d + pd.Timedelta(days=1), H) + pd.to_timedelta(3 * (steps - 1), unit="h"),
                    "observed_ap": y.ravel(), "geoi_ap_pred": seeds[s].ravel(), "swpc_kp_forecast": kp.ravel(),
                    "swpc_ap_converted": sa.ravel(), "source_file": np.repeat(sw_file.loc[d].to_numpy(), H),
                    "is_cme_issue_day": np.repeat(is_cme, H), "is_storm_3h_bin": (y >= STORM_AP).ravel(),
                    "coverage_cohort": f"matched_{h}h"}))
        sel_days = {"all": slice(None)} | {f"day{k}": slice((k - 1) * BINS, k * BINS) for k in range(1, H // BINS + 1)}
        for dname, sl in sel_days.items():
            yy = y[:, sl]
            for who, pr in (("geoi_ens", ens[:, sl]), ("swpc", sa[:, sl]), ("persistence", pers[:, sl])):
                if who != "geoi_ens" and grp != "full":
                    continue
                r = {**base, "target_day": dname, "forecast": who, **scores(yy, pr)}
                (rows_o if dname == "all" else rows_d).append(r)
            sm = [scores(yy, seeds[s][:, sl]) for s in range(3)]
            r = {**base, "target_day": dname, "forecast": "geoi_seedmean",
                 **{k: np.mean([m[k] for m in sm]) for k in ("n_bins", "mae", "rmse", "cc", "me")},
                 **{f"{k}_seed{s}": sm[s][k] for s in range(3) for k in ("mae", "cc", "me")}}
            (rows_o if dname == "all" else rows_d).append(r)
            # paired bootstrap GeoIndex (ensemble) − SWPC, 3-hourly, and daily Ap of the target day
            b3 = boot(yy, ens[:, sl], sa[:, sl], d, args.boot)
            rows_b.append({**base, "target_day": dname, "resolution": "3h", **b3,
                           **{f"d_mae_seed{s}": np.abs(seeds[s][:, sl] - yy).mean() - np.abs(sa[:, sl] - yy).mean() for s in range(3)}})
            if dname != "all":
                yd, gd, sd = yy.mean(1, keepdims=True), ens[:, sl].mean(1, keepdims=True), sa[:, sl].mean(1, keepdims=True)
                rows_b.append({**base, "target_day": dname, "resolution": "daily", **boot(yd, gd, sd, d, args.boot)})
        for s in range(1, H + 1):
            yy = y[:, s - 1]
            rows_s.append({**base, "lead_step_3h": s, "target_day": (s - 1) // BINS + 1, "forecast": "geoi_ens", **scores(yy, ens[:, s - 1])})
            if grp == "full":
                rows_s.append({**base, "lead_step_3h": s, "target_day": (s - 1) // BINS + 1, "forecast": "swpc", **scores(yy, sa[:, s - 1])})
        for dname, sl in sel_days.items():
            yy, ge, ss = y[:, sl], ens[:, sl], sa[:, sl]
            st = yy >= STORM_AP
            for subset, m in (("storm_bins", st), ("non_storm_bins", ~st), ("cme_issue_days", np.repeat(is_cme[:, None], yy.shape[1], 1))):
                for who, pr in (("geoi_ens", ge), ("swpc", ss)):
                    if who == "swpc" and grp != "full":
                        continue
                    rows_x.append({**base, "target_day": dname, "subset": subset, "forecast": who, **scores(yy[m], pr[m])})
        # daily derived (target days) with SWPC Kp-derived, official and RSGA
        for k in range(1, H // BINS + 1):
            sl = slice((k - 1) * BINS, k * BINS)
            yo = y[:, sl].mean(1)
            off = sw_off.loc[d, k].to_numpy(float)
            r_ = rsga[(rsga.kind == "predicted") & (rsga.lead == k)].drop_duplicates("target_date", keep="last")
            rg = r_.set_index(pd.DatetimeIndex(r_.target_date).as_unit("ns"))["ap"].reindex(d + pd.Timedelta(days=k)).to_numpy()
            for who, pr in (("geoi_derived", ens[:, sl].mean(1)), ("swpc_kp_derived", sa[:, sl].mean(1)),
                            ("swpc_official_predicted_ap", off), ("swpc_rsga", rg)):
                if who != "geoi_derived" and grp != "full":
                    continue
                ok = ~np.isnan(pr)
                rows_day.append({**base, "target_day": k, "forecast": who, "n_issue_dates_scored": int(ok.sum()),
                                 **{kk: v for kk, v in scores(yo[ok], pr[ok]).items() if kk != "n_bins"}})
    pd.concat(mt, ignore_index=True).to_parquet(out / "processed" / "matched_ap3h_24_48_72.parquet", index=False)
    pd.DataFrame(rows_o).to_csv(out / "metrics" / "metrics_3hourly_overall.csv", index=False)
    pd.DataFrame(rows_d).to_csv(out / "metrics" / "metrics_3hourly_by_target_day.csv", index=False)
    pd.DataFrame(rows_s).to_csv(out / "metrics" / "metrics_3hourly_by_step.csv", index=False)
    pd.DataFrame(rows_x).to_csv(out / "metrics" / "metrics_3hourly_storm_and_cme.csv", index=False)
    pd.DataFrame(rows_day).to_csv(out / "metrics" / "metrics_daily_derived.csv", index=False)
    pd.DataFrame(rows_b).to_csv(out / "metrics" / "paired_bootstrap_diff.csv", index=False)
    # daily cross-check of the SWPC products on the matched issue dates
    cx = []
    for k in (1, 2, 3):
        sl = slice((k - 1) * BINS, k * BINS)
        r_ = rsga[(rsga.kind == "predicted") & (rsga.lead == k)].drop_duplicates("target_date", keep="last")
        rg = r_.set_index(pd.DatetimeIndex(r_.target_date).as_unit("ns"))["ap"]
        cx.append(pd.DataFrame({"issue_date": matched.date, "target_day_k": k,
                                "swpc_kp_derived_daily_ap": sw_ap.loc[matched].to_numpy()[:, sl].mean(1),
                                "swpc_official_predicted_ap": sw_off.loc[matched, k].to_numpy(),
                                "swpc_rsga_ap": rg.reindex(matched + pd.Timedelta(days=k)).to_numpy()}))
    pd.concat(cx).to_csv(out / "processed" / "swpc_daily_ap_crosscheck.csv", index=False)
    print(f"issue dates {len(issue)}, with SWPC {len(matched)}, CME issue days matched {int(cme[issue.isin(sw_ap.index)].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
