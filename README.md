# geoindex-daily

Daily-cadence index forecasting, weeks ahead: 30 days of history → the next 60 daily
values. First target is **Ap** (planetary daily index); the index is a config choice, so
F10.7 / Kp / SN run through the same pipeline. Sibling of `geoindex-model`, which owns
the 30-min ap30/hp30, hours-ahead problem — the two share databases, not code.

Inputs are the index history (time-series encoder: MOMENT) and one solar image per day
(image encoder: Surya, frozen or LoRA). The operational yardstick is SWPC's 45-day Ap
forecast.

A second track (lead 1, `configs/ap_lead1.yaml`) forecasts daily Ap **one day ahead** from
1–7 days of Ap, GOES X-ray and OMNI solar wind (`l1`), and adds AIA 193 / AIA 211 / HMI /
LASCO C2 frames at four slots per day through a (2+1)D 3D CNN (`l2`). Results live in the
Research vault, not here.

## Layout

```
configs/ap.yaml            index, window lengths, rotation period, storm threshold, splits
configs/ap_1985.yaml       the same contract on the GFZ daily table, train 1985–2019
configs/ap_lead1.yaml      lead-1 track: features, lengths, slots, image channels, splits A / B, model settings
geoindex_daily/
  db.py                    read-only connection to the shared PostgreSQL (SOLARIS_DB_* env)
  daily_index.py           OMNI hourly → daily Ap/Kp/F10.7/SN/Dst table (parquet)
  baselines.py             climatology, persistence, 27-day recurrence (honest for h > 27)
  metrics.py               MAE, RMSE, log-MAE, corr, storm POD/FAR/CSI
  windows.py               (30-day input, 60-day target) samples and date splits
  swpc.py                  SWPC 45-day parsers; reader for the PRF outlook table loaded by solaris-data
  encoders/moment.py       MOMENT (frozen embedding / forecasting head) over 30-day windows
  encoders/surya.py        frozen Surya over SuryaBench (t-60, t) pairs → mean / first / G×G grid embeddings
  features.py              lead-1 daily features (Ap, XRS-B mean/max, OMNI daily solar wind)
  lead1.py                 lead-1 windows, common issue days, year splits, monthly block bootstrap
  slots.py                 per-day image slot table (nearest frame per slot × source within ±3 h)
  frames.py                frame preprocessing to uint8 1024² (AIA, HMI flip, LASCO C2 raw / running difference)
  image_data.py            frame-window dataset for the lead-1 image arms
  models/ts_cnn.py         1-D CNN over L days of daily features
  models/image_fusion.py   (2+1)D image branch + 1-D CNN branch, late fusion
  models/sinet.py          the co-author's SINet (loads their checkpoint strictly)
scripts/
  build_daily_index.py     DB → $GEOINDEX_DAILY_DATA/daily_index.parquet
  build_slot_table.py      DB → $GEOINDEX_DAILY_DATA/slot_table.parquet: nearest frame per day × slot (00/06/12/18 UT) × image source
  build_daily_features.py  daily_index + GOES XRS-B daily mean/max (log10) + OMNI daily solar wind → daily_features.parquet (lead-1 track)
  lead1_ts.py              lead-1 time-series arms (Ap · +X-ray · +solar wind · all) × ridge / 1-D CNN × L = 1..7, 14, 30; both splits; bootstrap (configs/ap_lead1.yaml)
  build_frame_cache.py     slot table → uint8 1024² frame cache per slot and channel (prev / calibrate / build / index); NAS needed
  lead1_image.py           lead-1 image arms (D: images + Ap · E: images + all · I: images only): (2+1)D branch + 1-D CNN branch, paired with l1
  lead1_l2_summary.py      l2 seed-mean forecasts vs their no-image l1 twins, the L = 1 curve and persistence; bootstrap → lead1/l2_summary_*.csv
  plot_lead1_figures.py    lead-1 report figures: length curve, image effect, case windows, arch_lead1_fusion.png
  eval_baselines.py        baseline scores by lead time → CSV
  load_swpc_prf.py         space_weather.swpc_prf_outlook (solaris-data) → local parquet
  eval_swpc.py             score SWPC outlooks vs observed Ap, with references on the same pairs
  ts_only.py               time-series-only models: ridge on raw window / MOMENT embedding (--tag for a second data source)
  build_daily_index_gfz.py GFZ Kp_ap_Ap_SN_F107_since_1932.txt → daily_index_gfz.parquet (1985–present; configs/ap_1985.yaml)
  sinet_train.py           SINet (co-author's TimesNet-style F10.7 model, geoindex_daily/models/sinet.py) on the daily windows, 3 seeds
  moment_finetune.py       MOMENT forecasting head (optionally encoder) fine-tuned on the windows
  compare_on_prf_issues.py like-for-like scores vs the SWPC outlook on the PRF issue dates
  extract_surya_embeddings.py  daily Surya embeddings from the archive tree → one npz per day
  fusion.py                stage-1 fusion: ridge on Ap history + PCA-reduced Surya token, with controls + bootstrap
  fusion_mlp.py            stage-2 fusion: small MLP heads, 3 seeds
  cache_surya_tokens.py    Surya token cache for LoRA training (GPU host)
  lora_train.py            Surya + LoRA (peft) fine-tuning with the grid head; lora_queue.sh runs the queue
  swpc_vs_null_models.py   SWPC's 27-day outlook decomposed against null models (selection vs damping)
  plot_architecture.py     architecture diagrams for the reports
  plot_report_figures.py   example-case and error-by-lead figures for the vault report
  render_report_pdf.py     vault markdown report (Obsidian embeds) → PDF via headless Edge/Chrome, else Playwright's Chromium
tests/                     pytest; no DB or NAS needed
```

Planned: a cycle-phase-balanced split for the 60-day track, a physics-shaped image probe
(coronal-hole area), a regularised 8×8 grid representation, MOMENT-large on the GPU host, the
channel ablation.

## Setup

```bash
conda activate geoindex-daily      # requirements.txt: torch, momentfm, transformers, requests, pypdf, ...
source ~/.solaris_env              # SOLARIS_DB_HOST/PORT/USER/PASSWORD (localhost on the on-site Mac)
export GEOINDEX_DAILY_DATA=~/Projects/GeoIndex/daily   # default; cloud-synced, not the NAS
```

## Commands

```bash
pytest                                                   # unit tests, offline
python scripts/build_daily_index.py                      # needs the DB, not the NAS
python scripts/build_slot_table.py [--check-files]        # image slot table; --check-files needs the NAS
python scripts/build_daily_features.py                   # lead-1 features; needs the DB
python scripts/lead1_ts.py --config configs/ap_lead1.yaml # lead-1 time-series arms (l1)
python scripts/build_frame_cache.py prev|calibrate|build|index   # frame cache (NAS; build runs on egghouse-gpu)
python scripts/lead1_image.py --arm E --lengths 1 2 3 5 7      # lead-1 image arms (l2), CUDA host
python scripts/lead1_l2_summary.py                       # l2 summary + bootstrap (after the sweep)
python scripts/plot_lead1_figures.py [--cases A:B ...]   # lead-1 report figures → vault experiments/figures
python scripts/render_report_pdf.py <report.md> ...      # PDF next to each vault report
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
- Lead-1 images: our own level-1 AIA / HMI archive and LASCO C2 (SuryaBench ends 2024), four
  slots per day (00/06/12/18 UT, `slot_table.parquet`), cached as uint8 1024² frames
  (`frames1024/`, ≈ 109 GB on the GPU host).

## What the baselines say (2011–2025, all issue dates)

27-day recurrence explains little of daily Ap on its own (r ≈ 0.2 linear, ≈ 0.3 in log),
and the climatological mean beats it on MAE. Recurrence skill is concentrated in the
declining phase (2016–2020). Whether a model "beats SWPC" therefore depends on the
score and the period — fix both before comparing. See `scripts/eval_baselines.py`.
