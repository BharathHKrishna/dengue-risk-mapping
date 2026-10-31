"""India state and district boundaries from the DataMeet community maps repository.

- States: ``States/Admin2.shp`` (current states and UTs, drawn to the
  Survey of India official external boundary, including Jammu and Kashmir and
  Ladakh as claimed by India).
- Districts: ``Districts/Census_2011/2011_Dist.shp`` (640 Census 2011
  districts, CC BY 2.5 India). District changes after 2011 are not reflected.

Source: https://github.com/datameet/maps . We use these instead of GAUL or
geoBoundaries because those draw de facto lines that misrepresent India's
official borders.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from dengue.utils.cache import download
from dengue.utils.config import path
from dengue.utils.log import get_logger

log = get_logger(__name__)

BASE = "https://raw.githubusercontent.com/datameet/maps/master/"
EXTS = (".shp", ".shx", ".dbf", ".prj")


def _fetch_shapefile(cfg: dict, rel: str) -> Path:
    raw = Path(cfg["paths"]["raw"])
    for ext in EXTS:
        download(BASE + rel + ext, raw / "datameet" / (rel + ext), raw, note="DataMeet maps")
    return raw / "datameet" / (rel + ".shp")


def normalize_name(s: str) -> str:
    """Lower-case, strip punctuation and the common '&'/'and' variants for name matching."""
    s = s.lower().replace("&", "and").replace("-", " ").replace(".", "")
    return " ".join(s.split())


def india_states(cfg: dict) -> gpd.GeoDataFrame:
    """States and UTs with columns state_id, state, geometry (EPSG:4326)."""
    out = path(cfg, "interim", "india_states.parquet")
    if out.exists():
        return gpd.read_parquet(out)
    g = gpd.read_file(_fetch_shapefile(cfg, "States/Admin2")).to_crs(4326)
    name_col = next(c for c in g.columns if c.lower() in ("st_nm", "state", "name"))
    g = g.rename(columns={name_col: "state"})[["state", "geometry"]]
    g = g.dissolve("state").reset_index()
    g.insert(0, "state_id", [f"IN-S{i:02d}" for i in range(len(g))])
    g.to_parquet(out)
    return g


def india_districts(cfg: dict) -> gpd.GeoDataFrame:
    """Census 2011 districts with columns district_id, district, state, geometry."""
    out = path(cfg, "interim", "india_districts.parquet")
    if out.exists():
        return gpd.read_parquet(out)
    g = gpd.read_file(_fetch_shapefile(cfg, "Districts/Census_2011/2011_Dist")).to_crs(4326)
    g = g.rename(columns={"DISTRICT": "district", "ST_NM": "state", "censuscode": "census_code"})
    g = g[["census_code", "district", "state", "geometry"]].reset_index(drop=True)
    g.insert(0, "district_id", [f"IN-D{int(c):03d}" for c in g["census_code"]])
    g.to_parquet(out)
    return g
