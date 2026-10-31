# Dengue risk mapping that is tested on places it has never seen

Monthly dengue risk from satellite climate, land and population data, trained
on Brazil's open weekly municipal case data, validated by holding out whole
regions, and transferred to India where only state-year case counts exist.

> **Research prototype.** The map shows modeled climate and environment based
> relative risk. It is validated against reported cases only for Brazilian
> municipalities and Indian state-years. It is not a case forecast.

**Live map:** _deployment pending_ &nbsp;|&nbsp; **API:** `GET /risk?region_id=BR-3304557&month=2025-03`

## Question

Can a model built from satellite, climate and population data predict dengue
incidence, and does it still work in a region it was not trained on?

## Data (all public, details and licenses in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md))

| Role | Source |
|---|---|
| Brazil labels | InfoDengue weekly notified cases, 5,570 municipalities, 2014 to 2025 |
| India labels | OpenDengue V1.3 state-year cases (from NCVBDC), 1997 to 2024 |
| Climate (monthly, lags 1 to 3) | ERA5-Land temperature, humidity, rainfall; CHIRPS rainfall; MODIS land surface temperature |
| Static | GHSL population and built surface, JRC surface water, MODIS NDVI, VIIRS night lights, SRTM elevation, ESA WorldCover |
| Boundaries | IBGE 2022 municipalities; DataMeet India states and Census 2011 districts (Survey of India external boundary) |

All Earth Engine extraction is in [src/dengue/ingest/gee_extract.py](src/dengue/ingest/gee_extract.py).

## Method

- **Unit:** municipality by month (Brazil). Weekly counts are assigned to the
  month of each epi week's Wednesday and scaled to a 4.345-week month.
- **Target:** `log1p(incidence per 100,000)`; alert target: month above the
  municipality's own 75th percentile (thresholds from training years only).
- **Features:** climate at lags 1, 2, 3 months; month of year (sine, cosine);
  static land and population covariates. A prediction for month `t` only uses
  data up to `t - 1` (tested in [tests/test_leakage.py](tests/test_leakage.py)).
- **Models:** seasonal climatology and last-year baselines; LightGBM without
  past cases (`lgbm_climate`, deployed) and with past cases (`lgbm_ar`,
  lags 1, 2, 3, 12); 3 seeds each.
- **Uncertainty:** LightGBM 5% and 95% quantile models, conformalized (CQR) on
  a held-out calibration year.
- **Validation:** temporal hold-out; spatial blocked CV holding out each of
  Brazil's 5 macro-regions (blocked in space and time); Brazil to India
  transfer by state-year rank. 95% CIs from a municipality cluster bootstrap.
- **Explainability:** SHAP (TreeExplainer), with biological sanity checks.
- **Tracking:** every run logged to MLflow (`mlruns/`) with config and git hash.

## Results

<!-- RESULTS -->

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Windows: .venv\Scripts\pip
earthengine authenticate && echo "EE_PROJECT=<your-ee-project>" > .env
python -m dengue.cli all          # or: make all
python scripts/make_tables.py     # renders results/tables.md
uvicorn app.api.main:app --port 8000   # map at http://localhost:8000
```

Stages (each cached, rerun from any point): `boundaries, cases, gee_brazil,
features, india, temporal, spatial, explain, final, transfer, export`.
Raw pulls are cached under `data/raw/` with URL and UTC pull time in
`data/raw/manifest.jsonl`. Dependencies are pinned in `pyproject.toml` and
`requirements.lock`. Full run on a laptop: about 1 hour of downloads
(InfoDengue is rate limited politely) plus about 1.5 hours of model fitting.

## Repository

```
configs/default.yaml   every setting (years, lags, splits, model params, seeds)
src/dengue/ingest      InfoDengue, OpenDengue, boundaries, Earth Engine
src/dengue/features    labels, lags, seasonality
src/dengue/models      baselines, LightGBM, conformal, deployment model
src/dengue/validation  splits (with leakage checks), metrics, bootstrap, transfer
src/dengue/explain     SHAP and sanity checks
app/                   FastAPI backend, Leaflet map, Dockerfile, exported data
.github/workflows      CI (ruff, pytest) and the monthly data refresh
docs/                  data sources, model card
report/                technical report
```

## Interview story

"I reused my thesis geospatial pipeline to build monthly dengue risk
predictions from climate, land and population data. I trained on Brazil's open
weekly municipal data, validated by holding out whole regions to test transfer
to unseen places, and then tested whether the model ranks Indian states
correctly against reported cases. I show uncertainty intervals and the drivers
behind each prediction, and the map updates monthly automatically. The main
lesson: a model that looks strong in random splits can degrade in new regions,
so spatial validation is essential before any public-health use."

<!-- STORY -->

## License and attribution

Code: MIT. Data remain under their own licenses (see
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)). Contains modified Copernicus
Climate Change Service information. India boundaries by the DataMeet India
community.
