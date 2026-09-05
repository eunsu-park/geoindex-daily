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
    example_plot(ap, dates, Y, P, sw, issue, out / f"example_{issue:%Y%m%d}.png")
    by_lead_plot(dates, Y, P, sw, out / "error_by_lead.png")
    print(f"issue {issue.date()} → {out}/example_{issue:%Y%m%d}.png, {out}/error_by_lead.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
