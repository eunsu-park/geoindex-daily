"""Lead-1 image arms (l2): (2+1)D image branch + time-series branch over L days of frames.

Arms: D = images + Ap · B = images + Ap + X-ray (the co-authors' input set) · E = images + all eight
daily features · I = images only. The issue
days, splits, target transform and references are exactly those of `lead1_ts.py` (same
config), so every score is paired with the l1 forecasts saved in `l1_preds_<split>.npz`.

    python scripts/lead1_image.py --arm E --lengths 2 --seeds 0 --split A            # one run
    python scripts/lead1_image.py --arm E --lengths 1 2 3 5 7 --seeds 0 1 2 --split A
    python scripts/lead1_image.py --arm E --lengths 2 --seeds 0 --limit 8 --epochs 1 --device mps  # smoke
Outputs in $GEOINDEX_DAILY_DATA/lead1/: l2_scores.csv (appended), l2_preds_<split>_<arm>_L<L>_s<seed>.npz,
l2_log_<split>_<arm>_L<L>_s<seed>.csv (per-epoch), checkpoints under lead1/ckpt/.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

# numpy's transparent-hugepage madvise on the large frame arrays drives the kernel into compaction
# stalls in the loader workers (3-19x slower frame loading on egghouse-gpu); must be set before numpy loads.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoindex_daily.daily_index import default_data_dir  # noqa: E402
from geoindex_daily.features import load_daily_features  # noqa: E402
from geoindex_daily.image_data import FrameWindowDataset, check_cache  # noqa: E402
from geoindex_daily.lead1 import (  # noqa: E402
    block_bootstrap_diff, common_issue_days, feature_windows, ffill_features,
    image_complete_days, score_lead1, split_masks,
)
from geoindex_daily.models.image_fusion import ImageFusion  # noqa: E402
from geoindex_daily.models.ts_cnn import default_device  # noqa: E402
from geoindex_daily.slots import load_slot_table  # noqa: E402

ARM_FEATURES = {"D": "A", "B": "B", "E": "E", "I": None}   # which feature list of the config feeds the TS branch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ap_lead1.yaml")
    p.add_argument("--arm", default="E", choices=list(ARM_FEATURES))
    p.add_argument("--split", default="A")
    p.add_argument("--lengths", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--channels", nargs="*", default=None, help="default from config image_channels")
    p.add_argument("--slots", nargs="*", type=int, default=None, help="slot hours (default from config)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="smoke test: use this many samples per split")
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--tag", default="l2")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    icfg = cfg.get("image_model", {})
    data = default_data_dir()
    out = data / cfg["out_dir"]
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    cache_dir = data / "frames1024"
    channels = tuple(args.channels or icfg.get("channels", ["aia193", "aia211", "hmi_m45", "lasco_c2"]))
    slot_hours = tuple(args.slots or cfg["slot_hours"])
    lengths = args.lengths or cfg["image_input_days"]
    seeds = args.seeds or cfg["seeds"]
    epochs = args.epochs or icfg.get("max_epochs", 40)
    patience = args.patience or icfg.get("patience", 8)
    batch = args.batch or icfg.get("batch_size", 8)
    lr = args.lr or icfg.get("lr", 3e-4)
    device = torch.device(args.device) if args.device else default_device()
    use_amp = (device.type == "cuda") and not args.no_amp
    thr = cfg["storm_threshold"]
    tf, inv = np.log1p, np.expm1

    # ---- samples: identical to lead1_ts.py ------------------------------------------------
    df = load_daily_features(data / cfg["data_file"])
    all_cols = sorted({c for a in cfg["features"] for c in cfg["features"][a]})
    df = ffill_features(df, all_cols, cfg["ffill_days"])
    df["ap_t"] = tf(df["ap"])
    slots = load_slot_table(data / cfg["slot_file"])
    img_days = image_complete_days(slots, cfg["image_sources"], tuple(cfg["slot_hours"]), cfg["sample_window_days"])
    days = common_issue_days(df, all_cols, cfg["history_days"], cfg["lead"], tuple(cfg["period"]), img_days)
    y_all = df["ap"].to_numpy()[df.index.get_indexer(days) + cfg["lead"]]
    masks = split_masks(days, cfg["splits"][args.split])
    feat_key = ARM_FEATURES[args.arm]
    ts_cols = [("ap_t" if c == "ap" else c) for c in cfg["features"][feat_key]] if feat_key else ["ap_t"]
    print(f"arm {args.arm} split {args.split} channels {channels} slots {slot_hours} device {device} amp {use_amp}; "
          f"{len(days)} issue days")

    l1 = out / f"l1_preds_{args.split}.npz"
    ref = np.load(l1, allow_pickle=True) if l1.exists() else None

    for L in lengths:
        X, yt = feature_windows(df, ts_cols, L, days, target_col="ap_t", lead=cfg["lead"])
        # the frames must exist for every sample; drop (and report) the ones not yet cached
        ok = check_cache(cache_dir, days, L, slot_hours, channels)["ok"].to_numpy()
        sel = {k: m & ok for k, m in masks.items()}
        if args.limit:                       # smoke test of the mechanics only: same few cached days everywhere
            idx = np.flatnonzero(ok)[: args.limit]
            for k in sel:
                sel[k] = np.zeros_like(ok); sel[k][idx] = True
        print(f"L={L}: cached {ok.sum()}/{len(ok)} issue days; train {sel['train'].sum()} val {sel['val'].sum()} "
              f"test {sel['test'].sum()}")
        if sel["train"].sum() == 0 or sel["val"].sum() == 0 or sel["test"].sum() == 0:
            print("  nothing to train on; skip"); continue
        tr, va, te = sel["train"], sel["val"], sel["test"]
        m, s = X[tr].reshape(-1, X.shape[-1]).mean(0), X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-8
        Xs = (X - m) / s
        ym, ys = yt[tr].mean(), yt[tr].std()
        yz = (yt - ym) / ys

        def loader(mask, shuffle, seed=0):
            ds = FrameWindowDataset(cache_dir, days[mask], L, slot_hours, channels, Xs[mask], yz[mask])
            g = torch.Generator().manual_seed(seed)
            return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=args.workers,
                              pin_memory=device.type == "cuda", generator=g, persistent_workers=args.workers > 0)

        dl_tr, dl_va, dl_te = loader(tr, True), loader(va, False), loader(te, False)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = ImageFusion(len(channels), len(ts_cols) if feat_key else 0,
                                widths=tuple(icfg.get("widths", (32, 64, 128, 256))),
                                ts_width=cfg["ts_cnn"]["width"], hidden=icfg.get("hidden", 128),
                                dropout=icfg.get("dropout", 0.2)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=icfg.get("weight_decay", 1e-4))
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            tag = f"{args.split}_{args.arm}_L{L}_s{seed}"
            logf = open(out / f"{args.tag}_log_{tag}.csv", "w", newline="")
            log = csv.writer(logf); log.writerow(["epoch", "train_loss", "val_mae", "seconds"])
            best, bad, best_ep = np.inf, 0, 0
            ckpt = out / "ckpt" / f"{args.tag}_{tag}.pt"

            def run_eval(dl):
                model.eval(); outp = []
                with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                    for img, ts, _ in dl:
                        img, ts = img.to(device, non_blocking=True), ts.to(device)
                        outp.append(model(img, ts if feat_key else None).float().cpu().numpy())
                return inv(np.concatenate(outp) * ys + ym)

            for ep in range(1, epochs + 1):
                model.train(); t0 = time.time(); tot = n = 0
                for img, ts, yb in dl_tr:
                    img, ts, yb = img.to(device, non_blocking=True), ts.to(device), yb.to(device)
                    with torch.autocast(device.type, enabled=use_amp):
                        loss = torch.nn.functional.mse_loss(model(img, ts if feat_key else None).float(), yb)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    tot += loss.item() * len(yb); n += len(yb)
                pv = run_eval(dl_va)
                vmae = float(np.abs(pv - inv(yt[va] )).mean())
                log.writerow([ep, tot / max(n, 1), vmae, round(time.time() - t0)]); logf.flush()
                print(f"  {tag} epoch {ep:3d} loss {tot / max(n, 1):.4f} val MAE {vmae:.3f} ({time.time() - t0:.0f}s)", flush=True)
                if vmae < best - 1e-4:
                    best, bad, best_ep = vmae, 0, ep
                    torch.save(model.state_dict(), ckpt)
                else:
                    bad += 1
                    if bad >= patience:
                        break
            logf.close()
            model.load_state_dict(torch.load(ckpt, map_location=device))
            pt = run_eval(dl_te)
            dte = days[te]
            sc = score_lead1(y_all[te], pt, dte, thr)
            row = {"split": args.split, "arm": args.arm, "model": "fusion", "L": L, "seed": seed,
                   "channels": "+".join(channels), "slots": len(slot_hours), "best_epoch": best_ep,
                   "val_mae": best, "n_train": int(tr.sum()), **sc}
            # paired comparisons with the l1 forecasts on the same days
            if ref is not None and not args.limit:
                rdates = pd.DatetimeIndex(ref["dates"])
                pos = rdates.get_indexer(dte)
                assert (pos >= 0).all(), "test days not in l1_preds"
                for name, key in (("vs_A_cnn_sameL", f"A_cnn_L{L}"), ("vs_B_cnn_sameL", f"B_cnn_L{L}"), ("vs_E_cnn_sameL", f"E_cnn_L{L}"),
                                  ("vs_E_cnn_L2", "E_cnn_L2"), ("vs_persistence", "persistence")):
                    if key in ref:
                        r = block_bootstrap_diff(y_all[te], pt, ref[key][pos], dte, cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"])
                        row.update({f"{name}_dmae": r["d_mae"], f"{name}_dmae_lo": r["d_mae_lo"], f"{name}_dmae_hi": r["d_mae_hi"],
                                    f"{name}_dcc": r["d_corr"], f"{name}_dcc_lo": r["d_corr_lo"], f"{name}_dcc_hi": r["d_corr_hi"]})
            np.savez(out / f"{args.tag}_preds_{tag}.npz", y=y_all[te], dates=dte.to_numpy(), pred=pt)
            sfile = out / f"{args.tag}_scores.csv"
            pd.DataFrame([row]).to_csv(sfile, mode="a", header=not sfile.exists(), index=False)
            print(f"== {tag}: test MAE {sc['mae']:.3f} CC {sc['corr']:.3f} (best epoch {best_ep}, val MAE {best:.3f})"
                  + (f"; vs E-cnn L2: dMAE {row.get('vs_E_cnn_L2_dmae', float('nan')):+.3f} dCC {row.get('vs_E_cnn_L2_dcc', float('nan')):+.3f}"
                     if "vs_E_cnn_L2_dmae" in row else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
