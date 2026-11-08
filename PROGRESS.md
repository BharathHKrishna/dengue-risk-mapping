# Progress

Resume rule: on restart, read this file first and continue from "Next step".
All pipeline stages: `python -m dengue.cli <stage>`, in order:
boundaries, cases, gee_brazil, features, india, temporal, spatial, explain, final, transfer, export
(`python -m dengue.cli all` runs everything). Earth Engine project is in `.env` (EE_PROJECT, gitignored).

## Current state (2026-11-08): stages spatial, explain, final, transfer, export, ablation running

Runs as a background chain in the Claude session (WMI-launched processes lost network after
sleep, so they are not used). If interrupted: check `logs/pipeline.log`, rerun the remaining
stages with `python -m dengue.cli <stage>` in order (everything before is cached).

## Week 2 report (done)
- InfoDengue: 5,570 municipalities, 3,492,390 weekly rows (627 weeks each, none missing).
- Processed table: 735,240 municipality-months, 2015-01 to 2025-12 (`python -m dengue.cli features`).
- Labels are notified cases (higher than "probable" totals; 2024: 10,545,135 notified).

## Week 3 report (done)
- Brazil monthly climate: 807,650 rows (5,570 x 145 months, 2014-09 to 2026-09); 3 island municipalities lack ERA5-Land.
- India: 641 districts, 92,945 climate rows; GHSL population sums to 1,399 M.

## Week 4 report (done): temporal hold-out (results/temporal_metrics.csv, `python -m dengue.cli temporal`)
- Climate-only LightGBM: skill vs climatology 0.012 [0.006, 0.017]; MAE 1.821 vs climatology 1.758.
- With past cases: skill 0.660 [0.654, 0.667], Spearman 0.821, alert AUC-PR 0.806.
- Test years far above training levels (mean y 2.23/3.27/2.60 vs 0.98 to 2.15); climate explains little of the jump.
- 90% conformal coverage: 0.83 (climate), 0.88 (AR). Under nominal because of the year shift.

## Week 1 report (done)
- Verified (2026-10-30/31): InfoDengue API, OpenDengue V1.3 zips, IBGE geoftp mesh, DataMeet India layers, all 10 GEE asset IDs.
- Not reachable: IBGE REST API (timeout from this machine). Using the geoftp mesh instead.
- OpenDengue India: state-year 1997 to 2024 (no 2013, 2014), no district or monthly state data; national monthly 2024 to 2025.
- Changed vs brief: VIIRS `ANNUAL_V22` starts in 2022 (not 2020); MLflow 3.17 needs a SQLite backend.
- Repo skeleton, pinned deps, config, cache manifest, CI, pre-commit.

## Week 2 (in progress)
- Boundaries: 5,570 municipalities (two IBGE lagoon pseudo-units dropped).
- InfoDengue pull running (`python -m dengue.cli cases`, log logs/cases_pull.log). Parser validated on 1,400 real files: 627 weeks each, no null cases or pop.
- Monthly label: epi weeks assigned to the month of their Wednesday; counts scaled to a 4.345-week month.

## Week 3 (in progress)
- GEE static covariates for Brazil: done (5,570 rows, GHSL nodata masked after a negative-density bug was caught).
- GEE monthly climate Brazil 2014-09 to 2026-09: running (logs/gee_brazil.log). India districts: running (logs/india.log).

## Weeks 4 to 9 (code done, waiting for data)
- experiments.py: climatology and last-year baselines, LightGBM with and without AR features, 3 seeds, cluster bootstrap CIs, CQR conformal intervals, alert metrics, MLflow logging.
- Spatial CV: leave-one-macro-region-out, blocked in space and time.
- SHAP + biological sanity checks, final deployment model, India transfer test, map export.
- FastAPI + Leaflet app, Dockerfile, render.yaml, monthly GitHub Actions update.
- 27 tests pass (parsers, lags, no-leakage, splits, metrics, conformal, API).
- Smoke test on a North-only subset ran every modeling stage end to end (numbers not reported, toy run).

## Open problems
- Smoke run hint: conformal intervals calibrated on 2022 may under-cover later years (shift). Check on full data.
- Deployment needs a Render account and a GitHub remote. Monthly job needs GEE service account secrets.
- Docker daemon is not running on this machine, so the image is not built locally yet.
- CLAUDE.md and the brief are in the first commit's history (untracked since). Strip before a public push if wanted.

## Next step
- When pulls finish: `features`, `temporal`, `spatial`, `explain`, `final`, `transfer`, `export`; then README, model card, report.
