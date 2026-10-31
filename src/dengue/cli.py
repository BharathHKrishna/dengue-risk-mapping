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


STAGES = {"boundaries": stage_boundaries, "cases": stage_cases, "gee_brazil": stage_gee_brazil}


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
