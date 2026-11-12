# Progress

Resume rule: on restart, read this file first and continue from "Next step".
All pipeline stages: `python -m dengue.cli <stage>`, in order:
boundaries, cases, gee_brazil, features, india, temporal, spatial, explain, final, transfer, export
(`python -m dengue.cli all` runs everything). Earth Engine project is in `.env` (EE_PROJECT, gitignored).

## Current state (2026-11-12): COMPLETE. Live at https://dengue-risk-map.onrender.com, repo github.com/BharathHKrishna/dengue-risk-mapping

## Weeks 5 to 7 report (done)
- Spatial CV (`spatial`): skill vs pooled climatology 0.357 CW, -0.046 N, 0.041 NE, 0.085 S, 0.287 SE; worse than local history except South (0.101). Coverage 0.74 to 0.98.
- SHAP (`explain`): temperature peak 25.8 to 27.5 C; rain and humidity positive; elevation top feature (plateau proxy).
- Ablation (`ablation`): with local climatology as input, lagged climate gives skill -0.008.
- India transfer (`transfer`): mean Spearman 0.146 vs density 0.258 vs persistence 0.695; seasonal timing Spearman 0.855 (2024).
- Diagnostic (`diag`, post hoc): no elevation gives India 0.199, Brazil skill -0.014.
- 24.7% of India district-months out of Brazil training range (`scripts/extra_analysis.py`).

## Weeks 8 to 10 report (done except deployment)
- FastAPI + Leaflet map tested locally and in Docker (image 662 MB), extrapolation warnings, incidence floored at 0.
- Monthly update job tested by simulation: matches full pipeline (corr 0.9999), writes out-of-range flags.
- README results + interview story, model card, report (report/report.md, about 2,500 words).
- 28 tests pass, ruff clean.

## Next step
- Optional: add EE service account secrets (EE_PROJECT, EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_JSON) so the monthly workflow refreshes the map.

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

## Open problems
- Conformal intervals under-cover in shifted years and some regions (confirmed on full data; reported).
- India state ranks: transfer failed (reported honestly; elevation proxy and out-of-range inputs).
- Deployment needs the GitHub repo, a Render account, and GEE service account secrets for the monthly job.
