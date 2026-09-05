"""Report figures: one example issue date, and MAE / CC by lead for every arm.

Style follows geoindex-model's validation plots (`src/plotting.py`): blue input, green
target with dots, red dashed prediction with x markers, grey dotted reference line, a
wheat score box. Reads the saved test forecasts of fusion stage 1 / 2 and the SWPC outlook
table; writes PNGs to --out (default: the vault's experiments/figures).

    python scripts/plot_report_figures.py --issue 2024-04-29
    python scripts/plot_report_figures.py            # picks the test Monday whose horizon holds the largest Ap
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402

VAULT_FIG = Path.home() / "Vaults/Research/GeoIndex/experiments/figures"
ARMS = {  # key in npz -> (label, colour, linestyle)
    "TS": ("Ridge, 30-day Ap", "tab:red", "--"),
    "PHASE": ("Ridge + F10.7/SN", "tab:purple", "--"),
    "TS_IMG7": ("Ridge + Surya token", "tab:brown", "--"),
    "climatology": ("Climatology", "gray", ":"),
    "recurrence27": ("Recurrence-27", "tab:olive", "-."),
}


def load_preds(d):
    s1 = np.load(d / "fusion_s1_ap_test_preds.npz")
    s2 = np.load(d / "fusion_s2_ap_test_preds.npz")
    dates = pd.to_datetime(s1["dates"])
    assert (pd.to_datetime(s2["dates"]) == dates).all()
    P = {k: s1[k] for k in ("TS", "PHASE", "TS_IMG7")}
    P["mlp_TS"] = s2["mlp_TS"]
    # climatology / recurrence come from the ts_only run (longer date range); align by date
    t1 = np.load(d / "ts_only_ap_test_preds.npz")
    idx = pd.Index(pd.to_datetime(t1["dates"])).get_indexer(dates)
    assert (idx >= 0).all()
    P["climatology"] = t1["climatology"][idx]
    P["recurrence27"] = t1["recurrence27"][idx]
    return dates, s1["y"], P


def swpc_on(dates, d):
    prf = pd.read_parquet(d / "swpc" / "prf_outlook.parquet")
    prf = prf[(prf.lead >= 1) & (prf.lead <= 26)]
    piv = prf.pivot_table(index="issue_date", columns="lead", values="ap")
    return piv.reindex(dates)  # NaN where SWPC issued nothing that day


def cc(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(y[m], p[m])[0, 1]


def example_plot(ap, dates, Y, P, sw, issue, out):
    i = int(np.flatnonzero(dates == issue)[0])
    t_in = np.arange(-29, 1); t_out = np.arange(1, 61)
    x_in = ap.loc[issue - pd.Timedelta(days=29): issue].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(t_in, x_in, "b-", linewidth=1.5, label="Input (daily Ap, 30 d)")
    ax.plot(t_out, Y[i], "g-o", linewidth=2, markersize=3, label="Target (observed)")
    for k in ("TS", "TS_IMG7"):
        lab, col, ls = ARMS[k]
        ax.plot(t_out, P[k][i], color=col, linestyle=ls, marker="x" if k == "TS" else None,
                markersize=4, linewidth=1.6, label=lab)
    ax.plot(t_out, P["climatology"][i], color="gray", linestyle=":", linewidth=1.2, label="Climatology")
    row = sw.loc[issue] if issue in sw.index else None
    if row is not None and row.notna().any():
        ax.plot(row.index.to_numpy(), row.to_numpy(), color="tab:orange", linestyle="-", linewidth=1.6,
                marker="s", markersize=3, label="SWPC 27-day outlook")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6, label="Issue day")
    ax.set_xlabel("Lead (days, relative to issue day)", fontsize=10)
    ax.set_ylabel("daily Ap", fontsize=10)
    ax.set_title(f"geoindex-daily @ {issue:%Y-%m-%d} — daily Ap, 30 d in / 60 d out", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    mae = np.abs(P["TS"][i] - Y[i]).mean(); r = cc(Y[i], P["TS"][i])
    mae2 = np.abs(P["TS_IMG7"][i] - Y[i]).mean(); r2 = cc(Y[i], P["TS_IMG7"][i])
    txt = f"Ridge  MAE {mae:.2f}  CC {r:.2f}\nRidge+Surya  MAE {mae2:.2f}  CC {r2:.2f}"
    if row is not None and row.notna().any():
        y26 = Y[i][:26]; s26 = row.to_numpy()[:26]
        txt += f"\nSWPC (1–26 d)  MAE {np.nanmean(np.abs(s26 - y26)):.2f}  CC {cc(y26, s26):.2f}"
    ax.text(0.99, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def by_lead_plot(dates, Y, P, sw, out):
    leads = np.arange(1, 61)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    series = {**{k: (ARMS[k][0], ARMS[k][1], ARMS[k][2], P[k]) for k in ARMS},
              "mlp_TS": ("MLP, 30-day Ap", "tab:cyan", "-", P["mlp_TS"])}
    for lab, col, ls, arr in series.values():
        mae = np.abs(arr - Y).mean(0)
        r = np.array([cc(Y[:, h], arr[:, h]) for h in range(60)])
        axes[0].plot(leads, mae, color=col, linestyle=ls, linewidth=1.6, label=lab)
        axes[1].plot(leads, r, color=col, linestyle=ls, linewidth=1.6, label=lab)
    # SWPC on the subset of test days that are PRF issue days, leads 1-26
    has = sw.notna().all(axis=1).to_numpy()
    if has.sum():
        Ys, Ss = Y[has][:, :26], sw.to_numpy()[has]
        Ts = P["TS"][has][:, :26]
        axes[0].plot(np.arange(1, 27), np.abs(Ss - Ys).mean(0), color="tab:orange", linewidth=2,
                     label=f"SWPC outlook ({has.sum()} Mondays)")
        axes[0].plot(np.arange(1, 27), np.abs(Ts - Ys).mean(0), color="tab:red", linewidth=1, alpha=0.5,
                     label="Ridge on the same Mondays")
        axes[1].plot(np.arange(1, 27), [cc(Ys[:, h], Ss[:, h]) for h in range(26)], color="tab:orange", linewidth=2,
                     label=f"SWPC outlook ({has.sum()} Mondays)")
        axes[1].plot(np.arange(1, 27), [cc(Ys[:, h], Ts[:, h]) for h in range(26)], color="tab:red", linewidth=1, alpha=0.5,
                     label="Ridge on the same Mondays")
    axes[0].set_ylabel("MAE (daily Ap)"); axes[1].set_ylabel("CC")
    axes[1].axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    for ax in axes:
        ax.set_xlabel("Lead (days)"); ax.grid(alpha=0.3); ax.set_xlim(1, 60)
    axes[0].legend(fontsize=8, loc="lower right"); axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Error by lead — test 2022–2024, {len(dates)} issue days (SWPC on its issue Mondays only)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def pick_cases(ap, dates, Y, sw):
    """Automatically choose one test Monday (with a SWPC outlook) per case type."""
    has = sw.notna().all(axis=1).to_numpy()
    cand = np.flatnonzero(has)
    mx, am = Y.max(1), Y.argmax(1) + 1  # peak value and its lead
    cases = {}
    used_peaks = []  # peak dates already shown, so the cases are distinct storms

    def best(mask, key):
        idx = cand[mask[cand]]
        for i in idx[np.argsort(-key[idx])]:
            peak = dates[i] + pd.Timedelta(days=int(am[i]))
            if all(abs((peak - u).days) > 5 for u in used_peaks):
                used_peaks.append(peak)
                return i
        return None
    cases["Storm at short lead (peak ≤ 7 d)"] = best((mx >= 40) & (am <= 7), mx)
    cases["Storm at mid lead (8–26 d)"] = best((mx >= 40) & (am >= 8) & (am <= 26), mx)
    cases["Storm at long lead (27–60 d)"] = best((mx >= 40) & (am >= 27), mx)
    cases["Quiet horizon (max Ap < 15)"] = best(mx < 15, -mx)
    # recurrent stream: a disturbed day in the input window that returns ~27 days later
    rec_score = np.full(len(dates), -np.inf)
    for i in cand:
        t = dates[i]
        x = ap.loc[t - pd.Timedelta(days=29): t].to_numpy()
        for k in range(1, 27):               # disturbed day at t-k, expected return at lead 27-k
            if x[29 - k] >= 30 and 1 <= 27 - k <= 60:
                rec_score[i] = max(rec_score[i], min(x[29 - k], Y[i][27 - k - 1]))
    cases["Recurrent stream (input storm returns ~27 d later)"] = best(np.isfinite(rec_score), rec_score)
    return {k: v for k, v in cases.items() if v is not None}


def cases_plot(ap, dates, Y, P, sw, cases, out_dir, slugs):
    """One PNG per case: `<out_dir>/case_<slug>_<issue>.png`. Returns the file list."""
    files = []
    for (title, i), slug in zip(cases.items(), slugs):
        fig, ax = plt.subplots(figsize=(14, 4.2))
        issue = dates[i]
        t_in = np.arange(-29, 1); t_out = np.arange(1, 61)
        x_in = ap.loc[issue - pd.Timedelta(days=29): issue].to_numpy()
        ax.plot(t_in, x_in, "b-", linewidth=1.5, label="Input (daily Ap, 30 d)")
        ax.plot(t_out, Y[i], "g-o", linewidth=2, markersize=3, label="Target (observed)")
        ax.plot(t_out, P["TS"][i], color="tab:red", linestyle="--", marker="x", markersize=4, linewidth=1.6, label="Ridge, 30-day Ap")
        ax.plot(t_out, P["TS_IMG7"][i], color="tab:brown", linestyle="--", linewidth=1.6, label="Ridge + Surya token")
        ax.plot(t_out, P["recurrence27"][i], color="tab:olive", linestyle="-.", linewidth=1.2, label="Recurrence-27")
        ax.plot(t_out, P["climatology"][i], color="gray", linestyle=":", linewidth=1.2, label="Climatology")
        row = sw.loc[issue]
        ax.plot(row.index.to_numpy(), row.to_numpy(), color="tab:orange", linestyle="-", marker="s", markersize=3, linewidth=1.6, label="SWPC 27-day outlook")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
        ax.set_ylabel("daily Ap", fontsize=10); ax.grid(alpha=0.3)
        ax.set_title(f"{title} — issue {issue:%Y-%m-%d} — daily Ap, 30 d in / 60 d out", fontsize=11, fontweight="bold")
        y26, s26 = Y[i][:26], row.to_numpy()[:26]
        txt = (f"Ridge  MAE {np.abs(P['TS'][i] - Y[i]).mean():.1f}  CC {cc(Y[i], P['TS'][i]):.2f}   |   "
               f"Ridge+Surya  MAE {np.abs(P['TS_IMG7'][i] - Y[i]).mean():.1f}  CC {cc(Y[i], P['TS_IMG7'][i]):.2f}   |   "
               f"SWPC 1–26 d  MAE {np.nanmean(np.abs(s26 - y26)):.1f}  CC {cc(y26, s26):.2f}")
        ax.text(0.99, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.set_xlabel("Lead (days, relative to issue day)", fontsize=10)
        f = out_dir / f"case_{slug}_{issue:%Y%m%d}.png"
        fig.savefig(f, dpi=120, bbox_inches="tight"); plt.close(fig)
        files.append(f)
    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--issue", default=None, help="issue date YYYY-MM-DD (default: auto-pick)")
    p.add_argument("--out", default=str(VAULT_FIG))
    args = p.parse_args()
    d = default_data_dir()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ap = load_daily()["ap"]
    dates, Y, P = load_preds(d)
    sw = swpc_on(dates, d)
    if args.issue:
        issue = pd.Timestamp(args.issue)
    else:  # the PRF Monday whose 60-day horizon holds the largest observed Ap
        has = sw.notna().all(axis=1).to_numpy()
        issue = dates[has][int(np.argmax(Y[has].max(1)))]
    if args.issue:
        example_plot(ap, dates, Y, P, sw, issue, out / f"example_{issue:%Y%m%d}.png")
    by_lead_plot(dates, Y, P, sw, out / "error_by_lead.png")
    cases = pick_cases(ap, dates, Y, sw)
    slugs = ["storm_short", "storm_mid", "storm_long", "quiet", "recurrent"][: len(cases)]
    files = cases_plot(ap, dates, Y, P, sw, cases, out, slugs)
    for (k, i), f in zip(cases.items(), files):
        print(f"  case: {k} → {dates[i].date()} (max Ap {Y[i].max():.0f} at lead {Y[i].argmax()+1}) → {f.name}")
    print(f"→ {out}/error_by_lead.png" + (f", example_{issue:%Y%m%d}.png" if args.issue else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
