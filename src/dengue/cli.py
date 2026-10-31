"""Pipeline entry point: ``python -m dengue.cli <stage> [--config PATH]``.

Stages run in order for ``all``. Each stage is cached, so re-running is cheap.
"""

from __future__ import annotations

import argparse

from dengue.utils.config import load_config
from dengue.utils.log import get_logger

log = get_logger("dengue.cli")


def stage_boundaries(cfg: dict) -> None:
    """Download and tidy Brazil municipal boundaries."""
    from dengue.ingest.boundaries import brazil_municipalities

    brazil_municipalities(cfg)


def stage_cases(cfg: dict) -> None:
    """Pull InfoDengue weekly cases for every municipality and build tables."""
    from dengue.ingest.boundaries import brazil_municipalities
    from dengue.ingest.infodengue import build_weekly_table, fetch_all

    geocodes = sorted(brazil_municipalities(cfg)["geocode"].tolist())
    failed = fetch_all(cfg, geocodes)
    if failed:
        log.warning("%d municipalities failed: %s", len(failed), failed[:20])
    weekly = build_weekly_table(cfg, geocodes)
    log.info("weekly table: %d rows, %d municipalities", len(weekly), weekly.geocode.nunique())


def stage_gee_brazil(cfg: dict) -> None:
    """Extract static and monthly climate covariates for Brazil municipalities."""
    from pathlib import Path

    import geopandas as gpd

    from dengue.ingest import gee_extract as g
    from dengue.utils.config import path

    g.initialize()
    gdf = gpd.read_parquet(path(cfg, "interim", "brazil_municipalities_simplified.parquet"))
    gdf = gdf[["geocode", "geometry"]]
    cache = Path(cfg["paths"]["raw"]) / "gee" / "brazil"
    gc = cfg["gee"]
    static = g.extract_static(gdf, "geocode", gc["static_year"], cache, ntl_year=gc["ntl_year"])
    static.to_parquet(path(cfg, "interim", "brazil_static.parquet"))
    log.info("static: %d rows", len(static))
    dyn = g.extract_dynamic(gdf, "geocode", gc["climate_start"], gc["climate_end"], cache)
    dyn.to_parquet(path(cfg, "interim", "brazil_climate_monthly.parquet"))
    log.info("climate: %d rows", len(dyn))


def stage_features(cfg: dict) -> None:
    """Join monthly labels, lagged climate and static covariates into the Brazil table."""
    import geopandas as gpd
    import pandas as pd

    from dengue.features.build import build_table
    from dengue.ingest.infodengue import weekly_to_monthly
    from dengue.utils.config import path

    weekly = pd.read_parquet(path(cfg, "interim", "infodengue_weekly.parquet"))
    monthly = weekly_to_monthly(weekly)
    clim = pd.read_parquet(path(cfg, "interim", "brazil_climate_monthly.parquet"))
    static = pd.read_parquet(path(cfg, "interim", "brazil_static.parquet"))
    muni = gpd.read_parquet(path(cfg, "interim", "brazil_municipalities.parquet"))
    df = build_table(monthly, clim, static.drop(columns=["pop_density"]), cfg)
    df = df.merge(muni[["geocode", "name", "uf", "region"]], on="geocode", how="left")
    df.to_parquet(path(cfg, "processed", "brazil_monthly.parquet"))
    log.info(
        "processed table: %d rows, %d municipalities, %s to %s",
        len(df),
        df["geocode"].nunique(),
        df["month"].min().date(),
        df["month"].max().date(),
    )


def stage_india(cfg: dict) -> None:
    """India boundaries, state-year labels, and GEE covariates for districts."""
    from pathlib import Path

    from dengue.ingest import gee_extract as g
    from dengue.ingest.india_boundaries import india_districts, india_states
    from dengue.ingest.opendengue import india_state_year
    from dengue.utils.config import path

    states = india_states(cfg)
    districts = india_districts(cfg)
    india_state_year(cfg, set(states["state"]))
    # Assign 2011 districts to current states (Telangana, Ladakh did not exist in 2011).
    pts = districts.copy()
    pts["geometry"] = pts.geometry.representative_point()
    joined = pts.sjoin(states[["state", "geometry"]], predicate="within", how="left")
    districts["state_current"] = joined["state_right"].to_numpy()
    districts["area_km2"] = districts.to_crs("ESRI:54009").area / 1e6
    simp = districts.copy()
    simp["geometry"] = simp.geometry.simplify(0.01, preserve_topology=True)
    simp.to_parquet(path(cfg, "interim", "india_districts_simplified.parquet"))
    g.initialize()
    gc = cfg["gee"]
    cache = Path(cfg["paths"]["raw"]) / "gee" / "india"
    fc = simp[["district_id", "geometry"]]
    static = g.extract_static(fc, "district_id", gc["static_year"], cache, ntl_year=gc["ntl_year"])
    static = static.merge(districts[["district_id", "area_km2"]], on="district_id")
    static["pop"] = static["pop_density"] * static["area_km2"]
    static.to_parquet(path(cfg, "interim", "india_static.parquet"))
    dyn = g.extract_dynamic(fc, "district_id", gc["climate_start"], gc["climate_end"], cache)
    dyn.to_parquet(path(cfg, "interim", "india_climate_monthly.parquet"))
    log.info("india: %d districts, %d climate rows", len(static), len(dyn))


def stage_temporal(cfg: dict) -> None:
    """Temporal hold-out validation."""
    from dengue.experiments import run_temporal

    print(run_temporal(cfg).to_string())


def stage_spatial(cfg: dict) -> None:
    """Spatial blocked cross-validation."""
    from dengue.experiments import run_spatial

    print(run_spatial(cfg).to_string())


def stage_explain(cfg: dict) -> None:
    """SHAP analysis and biological sanity checks."""
    from dengue.stages import run_explain

    run_explain(cfg)


def stage_final(cfg: dict) -> None:
    """Train the deployment model."""
    from dengue.stages import run_final

    run_final(cfg)


def stage_transfer(cfg: dict) -> None:
    """Brazil to India transfer test."""
    from dengue.stages import run_transfer

    run_transfer(cfg)


def stage_export(cfg: dict) -> None:
    """Export predictions and boundaries for the web app."""
    from dengue.stages import run_export

    run_export(cfg)


STAGES = {
    "boundaries": stage_boundaries,
    "cases": stage_cases,
    "gee_brazil": stage_gee_brazil,
    "features": stage_features,
    "india": stage_india,
    "temporal": stage_temporal,
    "spatial": stage_spatial,
    "explain": stage_explain,
    "final": stage_final,
    "transfer": stage_transfer,
    "export": stage_export,
}


def main() -> None:
    """Parse arguments and run the requested stage(s)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    for name in STAGES if args.stage == "all" else [args.stage]:
        log.info("stage %s", name)
        STAGES[name](cfg)


if __name__ == "__main__":
    main()
