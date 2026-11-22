"""Phase l6: the eight 3-hour ap of the next day from L days of 3-hour ap, X-ray and images, with uncertainty.

Plan: vault `planning/ap3h-plan.md` (co-authors' revised plan of 2026-11-06). A sample issued on day t (inputs
end at 24 UT of t, i.e. issued at 00 UT of t + 1) has

* time series: the 8L three-hour bins of days t−L+1 .. t, columns log1p ap, log10 X-ray 3-hour mean and maximum
  (`features_3h.parquet`, scripts/build_3h_features.py), standardised on the training days;
* images (`--model full`): HMI, AIA 193, AIA 211 at four frames a day and the LASCO C2 running difference at
  twelve a day with missing slots filled (config `lasco12f`), as in l5;
* target: log1p ap of the eight bins of day t + 1 (H = 8 · horizon days).

The head outputs a mean and a log-variance per bin (standardised log1p ap) and is trained with the Gaussian
negative log-likelihood; the point forecast of a bin is expm1(mean), the daily Ap forecast the mean of the eight.
Issue days are l5's (W41's 4,894, split A), minus any day whose 3-hour window or target is incomplete;
`--subset cme` keeps only the days with an Earth-directed DONKI CME (events/donki_cme.parquet) that started in
the 72 h before the issue time (days t−2, t−1, t). fp16 autocast on CUDA unless --no-amp.

Work order of 2026-11-22 (Models 0 / A / B): `--cme` adds the optional-CME residual (Model A) and `--footpoint` the
Solar-MACH footpoint residual on top (Model B; implies --cme), both from events/ap3h_cme_footpoint.parquet
(scripts/build_cme_footpoint_features.py) and geoindex_daily/models/optional_inputs.py. Model 0 is the run without
either flag. With --cme the test set is also scored with every CME slot emptied (stress test of missing CME data),
saved as <...>_cmeoff.npz beside the predictions.

    python scripts/ap3h_train.py --model full --lengths 1 --seeds 0 --split A --grad-checkpoint      # l6
    python scripts/ap3h_train.py --model ts --lengths 1 2 3 4 5 6 7                                  # l6-ts
    python scripts/ap3h_train.py --model full --subset cme --lengths 3 --seeds 0 1 2                 # l6-cme
    python scripts/ap3h_train.py --model full --lengths 7 --seeds 0 --limit 16 --epochs 1            # smoke
    python scripts/ap3h_train.py --model full --cme --lengths 4 --tag l7a                             # Model A
    python scripts/ap3h_train.py --model full --footpoint --lengths 4 --tag l7b                       # Model B

Outputs in --out-dir (default $GEOINDEX_DAILY_DATA/ap3h/): <tag>_preds_<split>_<model>_<subset>_L<L>_s<seed>.npz
(dates, y (N, H) ap, mu / sigma (N, H) in log1p ap), <tag>_log_*.csv (per epoch), <tag>_scores.csv, ckpt/.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")    # see lead1_image.py

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.ap3h import (BINS, CME_FEATURES, CME_SLOTS, FP_SPEEDS, TS_COLS, cme_days, gaussian_nll,  # noqa: E402
                                 optional_aux, scores, windows_3h)
from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.features import load_daily_features  # noqa: E402
from geoindex_daily.frames import cache_path  # noqa: E402
from geoindex_daily.image_data import FrameWindowDataset, TwoCadenceDataset, check_cache  # noqa: E402
from geoindex_daily.lead1 import common_issue_days, ffill_features, image_complete_days, split_masks  # noqa: E402
from geoindex_daily.models.image_fusion import ImageFusion  # noqa: E402
from geoindex_daily.models.optional_inputs import OptionalInputFusion  # noqa: E402
from geoindex_daily.models.ts_cnn import default_device  # noqa: E402
from geoindex_daily.slots import load_slot_table  # noqa: E402



class WithAux(torch.utils.data.Dataset):
    """(SDO, LASCO, ts, aux, y): a TwoCadenceDataset sample with the Models A / B input `aux` inserted."""

    def __init__(self, ds, aux: np.ndarray):
        assert len(ds) == len(aux)
        self.ds, self.aux = ds, torch.from_numpy(aux)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, las, ts, y = self.ds[i]
        return img, las, ts, self.aux[i], y


def l5_issue_days(cfg: dict, data: Path, cache_dir: Path, lcfg: dict) -> pd.DatetimeIndex:
    """The issue days of l5 (lead1_image.py --lasco12 --lasco-cfg lasco12f): W41's days, all slots usable."""
    df = load_daily_features(data / cfg["data_file"])
    all_cols = sorted({c for a in cfg["features"] for c in cfg["features"][a]})
    df = ffill_features(df, all_cols, cfg["ffill_days"])
    slots = load_slot_table(data / cfg["slot_file"])
    img_days = image_complete_days(slots, cfg["image_sources"], tuple(cfg["slot_hours"]), cfg["sample_window_days"])
    lslots = load_slot_table(data / lcfg["slot_file"])
    prev = pd.read_parquet(data / lcfg["prev_file"])[["slot_time", "prev_path"]]
    lslots = lslots.assign(slot_time=lslots.date + pd.to_timedelta(lslots.slot_hour, unit="h")).merge(
        prev, on="slot_time", how="left")
    lslots.loc[lslots.prev_path.isna(), "file_path"] = None
    cached = lslots.slot_time.map(lambda t: cache_path(cache_dir, lcfg["channel"], t).exists())
    lslots.loc[~cached, "file_path"] = None
    img_days = img_days.intersection(image_complete_days(
        lslots, ("lasco_c2",), tuple(lcfg["slot_hours"]), cfg["sample_window_days"]))
    return common_issue_days(df, all_cols, cfg["history_days"], cfg["lead"], tuple(cfg["period"]), img_days)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "ap_lead1.yaml"))
    p.add_argument("--model", choices=["full", "ts"], default="full")
    p.add_argument("--subset", choices=["all", "cme"], default="all")
    p.add_argument("--split", default="A")
    p.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--horizon-days", type=int, default=1, help="1 → 8 bins; 2, 3 → 48 / 72 h")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="smoke test: this many samples per split")
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--ts-pool", choices=["avg", "avg_last"], default="avg_last",
                   help="time-series branch pooling: avg_last (default) adds the last 3-hour step to the average; "
                        "avg alone dilutes the recent steps of a long window (l6-ts, 2026-11-07: daily-Ap CC "
                        "0.51 at L = 1 → 0.30 at L = 7)")
    p.add_argument("--cme", action="store_true", help="Model A: optional CME residual")
    p.add_argument("--footpoint", action="store_true", help="Model B: footpoint residual as well (implies --cme)")
    p.add_argument("--fp-shift-deg", type=float, default=0.0, help="phase-shift control: add this to every footpoint "
                   "longitude (0 for Model B)")
    p.add_argument("--cme-dropout", type=float, default=0.0, help="share of training samples whose CME slots are "
                   "emptied in each batch (work order §2.5; 0 = off)")
    p.add_argument("--tag", default="l6")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    args.cme = args.cme or args.footpoint
    if args.cme and args.model != "full":
        raise SystemExit("--cme / --footpoint are defined on the full model (Model 0 = --model full)")

    cfg = yaml.safe_load(Path(args.config).read_text())
    icfg, lcfg = cfg["image_model"], cfg["lasco12f"]
    data = default_data_dir()
    out = Path(args.out_dir) if args.out_dir else data / "ap3h"
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    cache_dir = data / "frames1024"
    channels = tuple(lcfg["sdo_channels"]) if args.model == "full" else ()
    slot_hours, lslot_hours = tuple(cfg["slot_hours"]), tuple(lcfg["slot_hours"])
    H = BINS * args.horizon_days
    epochs, patience = args.epochs or icfg["max_epochs"], args.patience or icfg["patience"]
    batch, lr = args.batch or icfg["batch_size"], args.lr or icfg["lr"]
    device = torch.device(args.device) if args.device else default_device()
    use_amp = (device.type == "cuda") and not args.no_amp

    # ---- samples ----------------------------------------------------------------------------
    f3 = pd.read_parquet(data / "features_3h.parquet")
    f3[["xray_mean_log", "xray_max_log"]] = f3[["xray_mean_log", "xray_max_log"]].ffill(limit=BINS * cfg["ffill_days"])
    f3["ap_t"] = np.log1p(f3["ap"])
    days = l5_issue_days(cfg, data, cache_dir, lcfg)
    n_l5 = len(days)
    X7, Y = windows_3h(f3, days, 7, H)
    complete = ~np.isnan(X7).any(axis=(1, 2)) & ~np.isnan(Y).any(axis=1)
    days = days[complete]
    masks = split_masks(days, cfg["splits"][args.split])
    if args.subset == "cme":
        cme = cme_days(days, data / "events" / "donki_cme.parquet")
        masks = {k: m & cme for k, m in masks.items()}
    print(f"model {args.model} subset {args.subset} split {args.split} H {H} device {device} amp {use_amp}; "
          f"{len(days)} issue days ({n_l5 - len(days)} of l5's {n_l5} lack 3-hour data); "
          + " ".join(f"{k} {int(m.sum())}" for k, m in masks.items()), flush=True)

    for L in args.lengths:
        X, Y = windows_3h(f3, days, L, H)
        # the image model's frames must exist; the time-series model uses the same days so the two pair up
        ok = (check_cache(cache_dir, days, L, slot_hours, lcfg["sdo_channels"])["ok"].to_numpy()
              & check_cache(cache_dir, days, L, lslot_hours, (lcfg["channel"],))["ok"].to_numpy())
        sel = {k: m & ok for k, m in masks.items()}
        train_all = sel["train"].copy()                  # before --limit: the aux standardisation's training days
        if args.limit:
            for k in sel:
                idx = np.flatnonzero(sel[k])[: args.limit]
                sel[k] = np.zeros_like(ok); sel[k][idx] = True
        tr, va, te = sel["train"], sel["val"], sel["test"]
        print(f"L={L}: train {tr.sum()} val {va.sum()} test {te.sum()}", flush=True)
        xm, xs = X[tr].reshape(-1, X.shape[-1]).mean(0), X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-8
        Xs = ((X - xm) / xs).astype(np.float32)
        ym, ys = Y[tr].mean(), Y[tr].std()
        Yz = ((Y - ym) / ys).astype(np.float32)
        aux = aux_off = None
        if args.cme:
            table = pd.read_parquet(data / "events" / "ap3h_cme_footpoint.parquet")
            aux = optional_aux(table, days, train_all, args.footpoint, args.fp_shift_deg)
            aux_off = aux.copy(); aux_off[:, CME_SLOTS * CME_FEATURES: CME_SLOTS * (CME_FEATURES + 1)] = 0
            st = table.set_index("issue_day").reindex(days).cme_status.to_numpy()
            print("  CME status " + "; ".join(f"{k}: " + ", ".join(f"{s} {int(((st == s) & m).sum())}" for s in
                  sorted(set(st))) for k, m in (("train", tr), ("val", va), ("test", te))), flush=True)

        def loader(mask, shuffle, seed=0, aux_arr=None):
            g = torch.Generator().manual_seed(seed)
            if args.model == "ts":
                ds = TensorDataset(torch.from_numpy(Xs[mask]), torch.from_numpy(Yz[mask]))
                return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g)
            ds = TwoCadenceDataset(
                FrameWindowDataset(cache_dir, days[mask], L, slot_hours, channels, Xs[mask], Yz[mask]),
                FrameWindowDataset(cache_dir, days[mask], L, lslot_hours, (lcfg["channel"],), Xs[mask], Yz[mask]))
            if aux_arr is not None:
                ds = WithAux(ds, aux_arr[mask])
            return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=args.workers,
                              pin_memory=device.type == "cuda", generator=g, persistent_workers=args.workers > 0)

        def unpack(b):
            """(SDO, LASCO, ts, aux, y) on the device from a 5-tuple (Models A / B), 4-tuple (images) or 2-tuple
            (time series) batch; aux is None without --cme."""
            if len(b) == 2:
                return None, None, b[0].to(device), None, b[1].to(device)
            a = b[3].to(device) if len(b) == 5 else None
            return (b[0].to(device, non_blocking=True), b[1].to(device, non_blocking=True),
                    b[2].to(device), a, b[-1].to(device))

        def forward(img, ts, las, a):
            return model(img, ts, las) if a is None else model(img, ts, las, a)

        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            kw = dict(widths=tuple(icfg["widths"]), ts_width=cfg["ts_cnn"]["width"], hidden=icfg["hidden"],
                      dropout=icfg["dropout"], lasco_channels=1 if args.model == "full" else 0,
                      grad_checkpoint=args.grad_checkpoint, out_dim=2 * H, ts_pool=args.ts_pool)
            if args.cme:
                model = OptionalInputFusion(len(channels), len(TS_COLS), cme_slots=CME_SLOTS, cme_features=CME_FEATURES,
                                            fp_features=2 * len(FP_SPEEDS) if args.footpoint else 0, **kw).to(device)
            else:
                model = ImageFusion(len(channels), len(TS_COLS), **kw).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=icfg["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            tag = f"{args.split}_{args.model}_{args.subset}_L{L}_s{seed}"
            dl_tr, dl_va, dl_te = loader(tr, True, seed, aux), loader(va, False, 0, aux), loader(te, False, 0, aux)
            logf = open(out / f"{args.tag}_log_{tag}.csv", "w", newline="")
            log = csv.writer(logf); log.writerow(["epoch", "train_nll", "val_mae_3h", "seconds"])
            best, bad, best_ep = np.inf, 0, 0
            ckpt = out / "ckpt" / f"{args.tag}_{tag}.pt"

            def run_eval(dl):
                """mu, sigma (N, H) in log1p ap."""
                model.eval(); outp = []
                with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                    for b in dl:
                        img, las, ts, a, _ = unpack(b)
                        outp.append(forward(img, ts, las, a).float().cpu().numpy())
                o = np.concatenate(outp)
                mu, lv = o[:, :H], np.clip(o[:, H:], -10, 10)
                return mu * ys + ym, np.exp(lv / 2) * ys

            for ep in range(1, epochs + 1):
                model.train(); t0 = time.time(); tot = n = 0
                for b in dl_tr:
                    img, las, ts, a, yb = unpack(b)
                    if a is not None and args.cme_dropout > 0:
                        drop = torch.rand(len(a), device=device) < args.cme_dropout
                        a = a.clone(); a[drop, CME_SLOTS * CME_FEATURES: CME_SLOTS * (CME_FEATURES + 1)] = 0
                    with torch.autocast(device.type, enabled=use_amp):
                        o = forward(img, ts, las, a)
                    loss = gaussian_nll(o, yb)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    tot += loss.item() * len(yb); n += len(yb)
                mu, _ = run_eval(dl_va)
                vmae = float(np.abs(np.expm1(mu) - np.expm1(Y[va])).mean())
                log.writerow([ep, tot / max(n, 1), vmae, round(time.time() - t0)]); logf.flush()
                print(f"  {tag} epoch {ep:3d} nll {tot / max(n, 1):.4f} val MAE(3h) {vmae:.3f} ({time.time() - t0:.0f}s)",
                      flush=True)
                if vmae < best - 1e-4:
                    best, bad, best_ep = vmae, 0, ep
                    torch.save(model.state_dict(), ckpt)
                else:
                    bad += 1
                    if bad >= patience:
                        break
            logf.close()
            model.load_state_dict(torch.load(ckpt, map_location=device))
            mu, sigma = run_eval(dl_te)
            y_te = np.expm1(Y[te])
            sc = scores(y_te, mu, sigma)
            row = {"split": args.split, "model": args.model, "subset": args.subset, "L": L, "seed": seed, "H": H,
                   "best_epoch": best_ep, "val_mae_3h": best, "n_train": int(tr.sum()), "n_test": int(te.sum()), **sc}
            np.savez(out / f"{args.tag}_preds_{tag}.npz", dates=days[te].to_numpy(), y=y_te, mu=mu, sigma=sigma)
            if args.cme:                                    # stress test: the same test days with no CME slot filled
                mu0, sg0 = run_eval(loader(te, False, 0, aux_off))
                np.savez(out / f"{args.tag}_preds_{tag}_cmeoff.npz", dates=days[te].to_numpy(), y=y_te, mu=mu0, sigma=sg0)
            sfile = out / f"{args.tag}_scores.csv"
            pd.DataFrame([row]).to_csv(sfile, mode="a", header=not sfile.exists(), index=False)
            print(f"== {tag}: 3h MAE {sc['mae_3h']:.3f} CC {sc['cc_3h']:.3f}; daily Ap MAE {sc['mae_day']:.3f} "
                  f"CC {sc['cc_day']:.3f}; 95 % cover {sc['cover95_3h']:.2f} (best epoch {best_ep})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
