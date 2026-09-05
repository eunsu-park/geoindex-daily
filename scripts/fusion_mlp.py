"""Stage-2 fusion (small nonlinear heads): Ap history + frozen Surya embeddings, 3 seeds.

Arms (fusion-plan §4, stage 2):
  TS            MLP on the 30 log1p-Ap values
  PHASE         MLP on TS + log F10.7 + SN
  TS+IMG7       TS  ‖ proj(7-day mean token)              proj = LayerNorm → Linear 1280→128 → GELU
  PHASE+IMG7    PHASE ‖ proj(7-day mean token)
  TS+STACK7     TS  ‖ Σ_d w_d · proj(token of day t−d), d = 0..6, w = softmax(learned)

Head: concat → Linear 256 → GELU → Dropout 0.1 → Linear 256 → GELU → Linear 60; MSE on
standardised log targets; AdamW (lr 1e-3, wd 1e-2); early stopping on val MAE (patience 6).
Seeds {0,1,2}; scores reported per seed and for the seed-mean forecast; monthly block
bootstrap of the CC gain (seed-mean forecasts) at leads 3-26 against the matched arm.

    python scripts/fusion_mlp.py --config configs/ap.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion import (GROUPS, SHOW_LEADS, TRANSFORMS, block_bootstrap_gain, daily_image_features,  # noqa: E402
                    load_embeddings, per_lead)
from geoindex_daily.daily_index import default_data_dir, load_daily  # noqa: E402
from geoindex_daily.windows import make_windows, split  # noqa: E402


def stack7(emb: pd.DataFrame, dates: pd.DatetimeIndex, ffill_days: int) -> np.ndarray:
    """(N, 7, 1280): tokens of days t-6..t (forward-filled; a still-missing day copies day t)."""
    full = pd.date_range(emb.index.min() - pd.Timedelta(days=6), emb.index.max(), freq="D")
    filled = emb.reindex(full).ffill(limit=ffill_days)
    out = np.empty((len(dates), 7, emb.shape[1]), dtype=np.float32)
    for j, t in enumerate(dates):
        block = filled.reindex(pd.date_range(t - pd.Timedelta(days=6), t, freq="D")).to_numpy()
        today = block[-1]
        for d in range(7):
            out[j, d] = block[d] if np.isfinite(block[d]).all() else today
    return out


class Proj(nn.Module):
    def __init__(self, d_in=1280, d_out=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU())

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """ts (B, n_ts) [+ img (B, 1280) | stack (B, 7, 1280)] → (B, horizon)."""

    def __init__(self, n_ts, horizon, img_mode=None, d_proj=128, hidden=256, dropout=0.1):
        super().__init__()
        self.img_mode = img_mode
        d = n_ts
        if img_mode in ("mean", "stack"):
            self.proj = Proj(1280, d_proj)
            d += d_proj
        if img_mode == "stack":
            self.day_logits = nn.Parameter(torch.zeros(7))
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon))

    def forward(self, ts, img=None):
        parts = [ts]
        if self.img_mode == "mean":
            parts.append(self.proj(img))
        elif self.img_mode == "stack":
            z = self.proj(img)                                   # (B, 7, d_proj)
            w = torch.softmax(self.day_logits, 0).view(1, 7, 1)
            parts.append((z * w).sum(1))
        return self.mlp(torch.cat(parts, dim=1))


def train_one(ts, img, Y, tr, va, img_mode, seed, device, epochs=60, patience=6, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(ts.shape[1], Y.shape[1], img_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    ts_t, Y_t = T(ts), T(Y)
    img_t = T(img) if img is not None else None
    tr_idx, va_idx = np.flatnonzero(tr), np.flatnonzero(va)

    def predict(idx):
        model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(idx), 512):
                j = idx[i: i + 512]
                out.append(model(ts_t[j], img_t[j] if img_t is not None else None))
            return torch.cat(out).cpu().numpy()

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), batch):
            j = tr_idx[i: i + batch]
            loss = nn.functional.mse_loss(model(ts_t[j], img_t[j] if img_t is not None else None), Y_t[j])
            opt.zero_grad(); loss.backward(); opt.step()
        vmae = np.abs(predict(va_idx) - Y[va]).mean()
        if vmae < best - 1e-4:
            best, bad = vmae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return predict(np.arange(len(Y))), best, ep + 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap.yaml")
    p.add_argument("--emb-dir", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cpu", help="cpu is fast enough for these heads")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--tag", default="fusion_s2")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fwd, inv = TRANSFORMS[cfg.get("ts_transform", "log1p")]
    L, H = cfg["input_days"], cfg["output_days"]
    d = load_daily()
    X, Y, dates = make_windows(fwd(d[cfg["index"]]), L, H)
    emb_dir = Path(args.emb_dir) if args.emb_dir else default_data_dir() / "surya" / "emb" / "13ch"
    emb = load_embeddings(emb_dir, "mean")
    img1, img7, ok, n_filled = daily_image_features(emb, dates, 3)
    phase = np.stack([np.log(d["f107"].reindex(dates).to_numpy()), d["sn"].reindex(dates).to_numpy()], axis=1)
    keep = ok & np.isfinite(phase).all(axis=1)
    X, Y, dates, img7, phase = X[keep], Y[keep], dates[keep], img7[keep], phase[keep]
    st7 = stack7(emb, dates, 3)
    masks = split(dates, cfg["splits"])
    tr, va, te = masks["train"], masks["val"], masks["test"]
    print(f"samples {len(dates)} (train {tr.sum()} val {va.sum()} test {te.sum()}); forward-filled {n_filled}")

    def z(a):  # standardise on train, per feature
        mu, sd = a[tr].mean(0), a[tr].std(0) + 1e-8
        return ((a - mu) / sd).astype(np.float32)

    ts_z, phase_z = z(X), z(np.concatenate([X, phase], axis=1))
    ym, ys = Y[tr].mean(), Y[tr].std()
    Yz = ((Y - ym) / ys).astype(np.float32)
    img7_f, st7_f = img7.astype(np.float32), st7.astype(np.float32)

    arms = {
        "TS": (ts_z, None, None),
        "PHASE": (phase_z, None, None),
        "TS+IMG7": (ts_z, img7_f, "mean"),
        "PHASE+IMG7": (phase_z, img7_f, "mean"),
        "TS+STACK7": (ts_z, st7_f, "stack"),
    }
    preds_seed, rows = {}, []
    for name, (ts, img, mode) in arms.items():
        P = []
        for s in args.seeds:
            t0 = time.time()
            Pz, vbest, ep = train_one(ts, img, Yz, tr, va, mode, s, args.device)
            Ps = Pz * ys + ym
            P.append(Ps)
            t = per_lead(Y[te], Ps[te], inv)
            r = {"arm": name, "seed": s, "epochs": ep, "val_mae_z": vbest}
            for g, (lo, hi) in GROUPS.items():
                r[f"mae_{g}"] = t.loc[lo:hi, "mae"].mean(); r[f"cc_{g}"] = t.loc[lo:hi, "corr"].mean()
            rows.append(r)
            print(f"{name:11s} seed {s}: {ep:2d} ep, val {vbest:.3f}, test MAE {r['mae_1-60']:.3f} "
                  f"CC {r['cc_1-60']:.3f} (3-26 {r['cc_3-26']:.3f}) [{time.time()-t0:.0f}s]")
        preds_seed[name] = np.stack(P)

    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby("arm")[[c for c in per_seed.columns if c.startswith(("mae_", "cc_"))]].agg(["mean", "std"])
    pd.set_option("display.width", 240)
    print("\nPer-seed mean ± std over lead groups (test):")
    print(agg.round(3).to_string())

    # seed-mean forecasts
    mean_tables = {n: per_lead(Y[te], P.mean(0)[te], inv) for n, P in preds_seed.items()}
    summ = pd.DataFrame({n: {f"{k}_{g}": t.loc[lo:hi, "mae" if k == "mae" else "corr"].mean()
                             for g, (lo, hi) in GROUPS.items() for k in ("mae", "cc")}
                         for n, t in mean_tables.items()}).T
    print("\nSeed-mean forecast, means over lead groups:")
    print(summ.round(3).to_string())
    show = pd.concat({n: t.loc[SHOW_LEADS] for n, t in mean_tables.items()}, axis=1)
    print("\nCC by lead (seed-mean):")
    print(show.xs("corr", axis=1, level=1).round(3).to_string())

    Yte = inv(Y[te])
    boot = []
    for a_, b_ in (("TS+IMG7", "TS"), ("TS+STACK7", "TS"), ("PHASE+IMG7", "PHASE"), ("PHASE", "TS")):
        for g, (lo, hi) in GROUPS.items():
            gain, cl, ch = block_bootstrap_gain(Yte, inv(preds_seed[a_].mean(0)[te]), inv(preds_seed[b_].mean(0)[te]),
                                                dates[te], (lo, hi), args.bootstrap)
            boot.append({"arm": a_, "vs": b_, "leads": g, "cc_gain": gain, "ci_lo": cl, "ci_hi": ch})
    boot = pd.DataFrame(boot)
    print(f"\nMonthly block bootstrap ({args.bootstrap}) of the CC gain, seed-mean forecasts:")
    print(boot.round(4).to_string(index=False))

    out = default_data_dir()
    per_seed.to_csv(out / f"{args.tag}_{cfg['index']}_per_seed.csv", index=False)
    summ.to_csv(out / f"{args.tag}_{cfg['index']}_summary.csv")
    boot.to_csv(out / f"{args.tag}_{cfg['index']}_bootstrap.csv", index=False)
    np.savez(out / f"{args.tag}_{cfg['index']}_test_preds.npz",
             dates=dates[te].strftime("%Y-%m-%d").to_numpy().astype("U10"), y=Yte,
             **{("mlp_" + n).replace("+", "_"): inv(P.mean(0)[te]) for n, P in preds_seed.items()})
    print(f"\nartifacts → {out}/{args.tag}_{cfg['index']}_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
