"""Administrative boundaries.

Brazil: IBGE 2022 municipal mesh (official, from geoftp.ibge.gov.br).
India: see ``dengue.ingest.india_boundaries``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from dengue.utils.cache import download
from dengue.utils.config import path
from dengue.utils.log import get_logger

log = get_logger(__name__)

LAGOON_CODES = (4300001, 4300002)


def region_of(geocode: pd.Series, regions: dict[int, str]) -> pd.Series:
    """Map 7-digit IBGE codes to macro-regions using the first digit."""
    return (geocode // 1_000_000).map(regions)


def brazil_municipalities(cfg: dict) -> gpd.GeoDataFrame:
    """Download (cached) and load Brazil municipalities as a tidy GeoDataFrame.

    Returns columns: geocode (int, 7-digit IBGE), name, uf, uf_code, region,
    area_km2, geometry (EPSG:4326). Also writes a simplified GeoParquet used
    by GEE extraction and the web map.
    """
    out = path(cfg, "interim", "brazil_municipalities.parquet")
    if out.exists():
        return gpd.read_parquet(out)
    raw = Path(cfg["paths"]["raw"])
    zpath = download(
        cfg["brazil"]["boundaries_url"],
        raw / "ibge" / "BR_Municipios_2022.zip",
        raw,
        timeout=600,
        note="IBGE malha municipal 2022",
    )
    exdir = raw / "ibge" / "BR_Municipios_2022"
    if not exdir.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(exdir)
    shp = next(exdir.rglob("*.shp"))
    src = gpd.read_file(shp).to_crs(4326)
    gdf = gpd.GeoDataFrame(
        {
            "geocode": src["CD_MUN"].astype(int),
            "name": src["NM_MUN"],
            "uf": src["SIGLA_UF"],
            "uf_code": src["CD_MUN"].astype(int) // 100_000,
            "area_km2": src["AREA_KM2"].astype(float),
        },
        geometry=src.geometry,
        crs=4326,
    )
    regions = {int(k): v for k, v in cfg["brazil"]["regions"].items()}
    gdf["region"] = region_of(gdf["geocode"], regions)
    # IBGE lists two lagoons (Lagoa dos Patos, Lagoa Mirim) as pseudo-municipalities.
    gdf = gdf[~gdf["geocode"].isin(LAGOON_CODES)].reset_index(drop=True)
    gdf.to_parquet(out)
    # Simplified copy (about 1 km tolerance) for GEE uploads and the web map.
    simp = gdf.copy()
    simp["geometry"] = simp.geometry.simplify(0.01, preserve_topology=True)
    simp.to_parquet(path(cfg, "interim", "brazil_municipalities_simplified.parquet"))
    log.info("loaded %d Brazil municipalities", len(gdf))
    return gdf
