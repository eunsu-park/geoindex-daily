# geoindex-daily

Daily-cadence index forecasting, weeks ahead: 30 days of history → the next 60 daily
values. First target is **Ap** (planetary daily index); the index is a config choice, so
F10.7 / Kp / SN run through the same pipeline. Sibling of `geoindex-model`, which owns
the 30-min ap30/hp30, hours-ahead problem — the two share databases, not code.

Inputs are the index history (time-series encoder: MOMENT) and one solar image per day
(image encoder: Surya, frozen or LoRA). The operational yardstick is SWPC's 45-day Ap
forecast.

## Layout

```
configs/ap.yaml            index, window lengths, rotation period, storm threshold, splits
geoindex_daily/
  db.py                    read-only connection to the shared PostgreSQL (SOLARIS_DB_* env)
  daily_index.py           OMNI hourly → daily Ap/Kp/F10.7/SN/Dst table (parquet)
  baselines.py             climatology, persistence, 27-day recurrence (honest for h > 27)
  metrics.py               MAE, RMSE, log-MAE, corr, storm POD/FAR/CSI
  windows.py               (30-day input, 60-day target) samples and date splits
  swpc.py                  SWPC forecast parsers: 45-day JSON/text, weekly PRF 27-day outlook
  encoders/moment.py       MOMENT (frozen embedding / forecasting head) over 30-day windows
  encoders/surya.py        frozen Surya over SuryaBench (t-60, t) pairs → mean / first / G×G grid embeddings
scripts/
  build_daily_index.py     DB → $GEOINDEX_DAILY_DATA/daily_index.parquet
  eval_baselines.py        baseline scores by lead time → CSV
  fetch_swpc_prf.py        mirror + parse the weekly PRF outlooks (NCEI, 1997–) → parquet
  eval_swpc.py             score SWPC outlooks vs observed Ap, with references on the same pairs
  ts_only.py               time-series-only models: ridge on raw window / MOMENT embedding
  moment_finetune.py       MOMENT forecasting head (optionally encoder) fine-tuned on the windows
  compare_on_prf_issues.py like-for-like scores vs the SWPC outlook on the PRF issue dates
  extract_surya_embeddings.py  daily Surya embeddings from the archive tree → one npz per day
tests/                     pytest; no DB or NAS needed
```

Planned: fusion models (MOMENT + Surya embeddings), the channel ablation for real-time
deployment, LoRA on a reduced setup.

## Setup

```bash
conda activate geoindex-daily      # requirements.txt: torch, momentfm, transformers, requests, pypdf, ...
source ~/.solaris_env              # SOLARIS_DB_HOST/PORT/USER/PASSWORD
export GEOINDEX_DAILY_DATA=~/Projects/GeoIndex/daily   # default; cloud-synced, not the NAS
```

## Commands

```bash
pytest                                                   # unit tests, offline
python scripts/build_daily_index.py                      # needs the DB, not the NAS
python scripts/eval_baselines.py --config configs/ap.yaml            # all issue dates
python scripts/eval_baselines.py --config configs/ap.yaml --split test
```

## Data

- Daily Ap = mean of the eight 3-hourly ap values. OMNI's hourly table repeats each 3-h
  ap on its three hours, so the 24-hour mean is exactly that. F10.7 and SN are daily in
  OMNI already. Coverage 2010-01 .. 2025-12 with no fills (checked 2026-09-02).
- Images: `solar_images.suryabench` (13-channel 4096² NetCDF, one file per timestamp),
  mirrored and registered by `solaris-data`. The daily plan is one frame per day at
  00 UT plus its (t−60 min) partner, because Surya's pretrained input is the pair.

## What the baselines say (2011–2025, all issue dates)

27-day recurrence explains little of daily Ap on its own (r ≈ 0.2 linear, ≈ 0.3 in log),
and the climatological mean beats it on MAE. Recurrence skill is concentrated in the
declining phase (2016–2020). Whether a model "beats SWPC" therefore depends on the
score and the period — fix both before comparing. See `scripts/eval_baselines.py`.
