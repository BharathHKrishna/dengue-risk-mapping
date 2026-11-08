"""Monthly incremental update for the deployed map.

Runs in GitHub Actions without the local raw-data cache. It uses only files
committed to the repo:

- ``app/data/*.geojson`` (boundaries, used as Earth Engine regions),
- ``app/data/*_static.parquet`` (static covariates),
- ``app/models/*`` (the frozen deployment model),

then pulls climate for the newest months from Earth Engine, predicts every
month that is not yet in the app tables, appends them and updates
``app/data/meta.json``. The model itself is never retrained here.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

from dengue.ingest import gee_extract as g
from dengue.models.final import build_unlabeled, load_final, ood_lists, predict_final
from dengue.stages import APP_DATA, _drivers, _round
from dengue.utils.config import ROOT, load_config
from dengue.utils.log import get_logger

log = get_logger(__name__)
APP_MODELS = ROOT / "app" / "models"


def latest_climate_month() -> str:
    """Latest month for which ERA5-Land monthly, CHIRPS and MODIS LST are all complete."""
    era = ee.ImageCollection(g.ASSETS["era5"]).aggregate_max("system:time_start")
    era_m = pd.Timestamp(ee.Date(era).format("YYYY-MM-dd").getInfo()).to_period("M")
    chirps = ee.ImageCollection(g.ASSETS["chirps"]).aggregate_max("system:time_start")
    ch_d = pd.Timestamp(ee.Date(chirps).format("YYYY-MM-dd").getInfo())
    ch_m = ch_d.to_period("M") if ch_d.is_month_end else ch_d.to_period("M") - 1
    return str(min(era_m, ch_m))


def _update_view(cfg, models, geo_file, static_file, id_col, pred_file, new_months, clim_start):
    gdf = gpd.read_file(APP_DATA / geo_file).rename(columns={"region_id": id_col})
    static = pd.read_parquet(APP_DATA / static_file)
    with tempfile.TemporaryDirectory() as tmp:
        clim = g.extract_dynamic(
            gdf[[id_col, "geometry"]],
            id_col,
            clim_start,
            str(new_months[-1] - 1),
            Path(tmp),
            workers=4,
        )
    X = build_unlabeled(clim, static, cfg, id_col, str(new_months[0]), str(new_months[-1]))
    pred, contrib = predict_final(models, X)
    out = pd.concat([X[[id_col, "month"]], pred], axis=1).rename(columns={id_col: "region_id"})
    out["drivers"] = _drivers(contrib, models["cols"])
    old = pd.read_parquet(APP_DATA / pred_file)
    if "rel_risk_pct" in old:
        out["rel_risk_pct"] = out.groupby("month")["pred"].rank(pct=True) * 100
        ranges = json.loads((APP_MODELS / "feature_ranges.json").read_text())
        out["ood"] = ood_lists(X, models["cols"], ranges)
    else:
        out["source"] = "final model"
        for c in ("obs", "cases", "pop"):
            out[c] = np.nan
    merged = pd.concat([old[~old["month"].isin(out["month"])], _round(out)], ignore_index=True)
    merged.to_parquet(APP_DATA / pred_file, index=False)
    log.info("%s: added %d rows", pred_file, len(out))


def run_update() -> None:
    """Predict any months newly made available by fresh climate data."""
    cfg = load_config()
    g.initialize()
    meta = json.loads((APP_DATA / "meta.json").read_text())
    have = pd.Period(meta["latest_prediction_month"], "M")
    latest_clim = pd.Period(latest_climate_month(), "M")
    target = latest_clim + 1
    if target <= have:
        log.info("no new climate month (latest %s, have predictions to %s)", latest_clim, have)
        return
    new_months = pd.period_range(have + 1, target, freq="M")
    lag_max = max(cfg["features"]["lags"])
    clim_start = str(new_months[0] - lag_max)
    # Load the frozen deployment model from app/models.
    cfg["paths"]["results"] = str(ROOT / "app")
    models = load_final(cfg)
    _update_view(
        cfg,
        models,
        "brazil_municipalities.geojson",
        "brazil_static.parquet",
        "geocode_id",
        "brazil_predictions.parquet",
        new_months,
        clim_start,
    )
    _update_view(
        cfg,
        models,
        "india_districts.geojson",
        "india_static.parquet",
        "district_id",
        "india_district_predictions.parquet",
        new_months,
        clim_start,
    )
    meta["climate_data_through"] = str(latest_clim)
    meta["latest_prediction_month"] = str(target)
    meta["generated_utc"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    (APP_DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("update done: %s", meta)


def copy_models_to_app(cfg: dict) -> None:
    """Copy the frozen deployment model files into ``app/models`` (committed)."""
    src = Path(cfg["paths"]["results"]) / "models"
    dst = APP_MODELS
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("final_*"):
        shutil.copy2(f, dst / f.name)


if __name__ == "__main__":
    run_update()
