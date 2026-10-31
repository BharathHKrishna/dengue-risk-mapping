"""Google Earth Engine extraction of climate and environment covariates per admin unit.

Authentication: run ``earthengine authenticate`` once, and set the environment
variable ``EE_PROJECT`` to a Cloud project registered for Earth Engine. For CI,
set ``EE_SERVICE_ACCOUNT`` and ``EE_PRIVATE_KEY_JSON`` (key file contents).
Nothing secret is stored in the repo.

All requests are cached per (unit chunk, year) under ``data/raw/gee/`` so a
failed or interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

from dengue.utils.log import get_logger

log = get_logger(__name__)

# Asset IDs, verified in the catalog by ``verify_assets``.
ASSETS = {
    "era5": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    "chirps": "UCSB-CHG/CHIRPS/DAILY",
    "lst": "MODIS/061/MOD11A2",
    "ndvi": "MODIS/061/MOD13Q1",
    "gsw": "JRC/GSW1_4/GlobalSurfaceWater",
    "ghsl_pop": "JRC/GHSL/P2023A/GHS_POP",
    "ghsl_built": "JRC/GHSL/P2023A/GHS_BUILT_S",
    "worldcover": "ESA/WorldCover/v200",
    "viirs": "NOAA/VIIRS/DNB/ANNUAL_V22",
    "srtm": "USGS/SRTMGL1_003",
}

# ESA WorldCover class codes -> static feature names.
WORLDCOVER = {
    10: "lc_tree",
    20: "lc_shrub",
    30: "lc_grass",
    40: "lc_crop",
    50: "lc_built",
    60: "lc_bare",
    80: "lc_water",
    90: "lc_wetland",
}

DYNAMIC_SCALE = 5000  # m. ERA5-Land is about 11 km, CHIRPS about 5.5 km.
STATIC_SCALE = 1000
WORLDCOVER_SCALE = 300


def initialize() -> None:
    """Initialize Earth Engine from env vars (service account if present, else user token)."""
    project = os.environ.get("EE_PROJECT")
    if not project:
        raise RuntimeError("set EE_PROJECT to your Earth Engine Cloud project id")
    sa, key = os.environ.get("EE_SERVICE_ACCOUNT"), os.environ.get("EE_PRIVATE_KEY_JSON")
    if sa and key:
        creds = ee.ServiceAccountCredentials(sa, key_data=key)
        ee.Initialize(creds, project=project)
    else:
        ee.Initialize(project=project)


def verify_assets() -> dict[str, str]:
    """Check every asset ID exists and return its catalog type. Raises on a missing asset."""
    out = {}
    for k, a in ASSETS.items():
        info = ee.data.getAsset(a)
        out[k] = info["type"]
    return out


def monthly_image(month: str) -> ee.Image:
    """Multi-band image of climate for one month (``YYYY-MM``).

    Bands: t2m_mean, t2m_min, t2m_max (deg C), rh_mean (percent, from 2 m
    temperature and dewpoint via the Magnus formula), precip_era5 (mm),
    precip_chirps (mm), lst_day (deg C, MODIS daytime LST).
    """
    start = ee.Date(month + "-01")
    end = start.advance(1, "month")
    era = ee.ImageCollection(ASSETS["era5"]).filterDate(start, end).first()
    t = era.select("temperature_2m").subtract(273.15)
    td = era.select("dewpoint_temperature_2m").subtract(273.15)

    def es(x: ee.Image) -> ee.Image:
        return x.multiply(17.625).divide(x.add(243.04)).exp()

    rh = es(td).divide(es(t)).multiply(100).rename("rh_mean")
    chirps = (
        ee.ImageCollection(ASSETS["chirps"]).filterDate(start, end).select("precipitation").sum()
    )
    lst = (
        ee.ImageCollection(ASSETS["lst"])
        .filterDate(start, end)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
    )
    return ee.Image.cat(
        [
            t.rename("t2m_mean"),
            era.select("temperature_2m_min").subtract(273.15).rename("t2m_min"),
            era.select("temperature_2m_max").subtract(273.15).rename("t2m_max"),
            rh,
            era.select("total_precipitation_sum").multiply(1000).rename("precip_era5"),
            chirps.rename("precip_chirps"),
            lst.rename("lst_day"),
        ]
    ).set("month", month)


def static_image(year: int, ntl_year: int = 2022) -> ee.Image:
    """Static covariates. Densities and fractions are per-pixel, so a mean reducer is exact.

    GHSL layers are on an equal-area 100 m Mollweide grid (0.01 km2 per cell),
    so persons per km2 = count * 100 and built fraction = m2 / 10000. Using
    ``pixelArea()`` instead would be evaluated at the reduction scale and be wrong.
    """
    # Negative values are GHSL nodata and must be masked before averaging.
    pop_raw = ee.Image(f"{ASSETS['ghsl_pop']}/{year}").select("population_count")
    built_raw = ee.Image(f"{ASSETS['ghsl_built']}/{year}").select("built_surface")
    pop = pop_raw.updateMask(pop_raw.gte(0)).multiply(100)
    built = built_raw.updateMask(built_raw.gte(0)).divide(1e4)
    water = ee.Image(ASSETS["gsw"]).select("occurrence").unmask(0).divide(100)
    ndvi = (
        ee.ImageCollection(ASSETS["ndvi"])
        .filterDate(f"{year - 1}-01-01", f"{year + 2}-01-01")
        .select("NDVI")
        .mean()
        .multiply(0.0001)
    )
    ntl = (
        ee.ImageCollection(ASSETS["viirs"])
        .filterDate(f"{ntl_year}-01-01", f"{ntl_year + 1}-01-01")
        .select("average")
        .first()
    )
    elev = ee.Image(ASSETS["srtm"]).select("elevation")
    return ee.Image.cat(
        [
            pop.rename("pop_density"),
            built.rename("built_frac"),
            water.rename("water_occurrence"),
            ndvi.rename("ndvi_mean"),
            ntl.rename("ntl_mean"),
            elev.rename("elevation"),
        ]
    )


def landcover_image() -> ee.Image:
    """One 0/1 band per WorldCover class; the mean over a unit is its area share."""
    wc = ee.ImageCollection(ASSETS["worldcover"]).first().select("Map")
    return ee.Image.cat([wc.eq(c).rename(n) for c, n in WORLDCOVER.items()])


def to_fc(gdf: gpd.GeoDataFrame, id_col: str) -> ee.FeatureCollection:
    """Client-side GeoDataFrame chunk to an ee.FeatureCollection with only the id property."""
    gj = json.loads(gdf[[id_col, "geometry"]].to_json())
    return ee.FeatureCollection(gj["features"])


def _reduce(img: ee.Image, fc: ee.FeatureCollection, scale: int) -> list[dict]:
    """Mean of every band over every feature; returns property dicts (no geometry)."""
    red = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale, tileScale=4)
    red = red.map(lambda f: ee.Feature(None, f.toDictionary()))
    return [f["properties"] for f in red.getInfo()["features"]]


def _reduce_with_fallback(
    img: ee.Image, gdf: gpd.GeoDataFrame, id_col: str, scale: int, bands: list[str]
) -> pd.DataFrame:
    """Reduce over polygons, then re-sample units too small to contain a pixel at their centroid."""
    df = pd.DataFrame(_reduce(img, to_fc(gdf, id_col), scale))
    for b in bands:
        if b not in df:
            df[b] = np.nan
    bad = df[df[bands].isna().all(axis=1)][id_col]
    if len(bad):
        pts = gdf[gdf[id_col].isin(bad)].copy()
        pts["geometry"] = pts.geometry.representative_point()
        fix = pd.DataFrame(_reduce(img, to_fc(pts, id_col), scale))
        if len(fix):
            df = pd.concat([df[~df[id_col].isin(bad)], fix], ignore_index=True)
    return df


def _months(year: int, start: str, end: str) -> list[str]:
    ms = pd.period_range(max(f"{year}-01", start), min(f"{year}-12", end), freq="M")
    return [str(m) for m in ms]


def extract_dynamic(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    start: str,
    end: str,
    cache_dir: Path,
    chunk: int = 150,
    workers: int = 6,
) -> pd.DataFrame:
    """Monthly climate per unit for months in ``[start, end]`` (YYYY-MM), cached per chunk-year."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf = gdf.sort_values(id_col).reset_index(drop=True)
    chunks = [gdf.iloc[i : i + chunk] for i in range(0, len(gdf), chunk)]
    years = range(int(start[:4]), int(end[:4]) + 1)
    bands = ["t2m_mean", "t2m_min", "t2m_max", "rh_mean", "precip_era5", "precip_chirps", "lst_day"]

    def job(ci: int, year: int) -> Path:
        f = cache_dir / f"dyn_{ci:03d}_{year}.parquet"
        if f.exists():
            return f
        rows = []
        for m in _months(year, start, end):
            df = _reduce_with_fallback(monthly_image(m), chunks[ci], id_col, DYNAMIC_SCALE, bands)
            df["month"] = m
            rows.append(df)
        out = pd.concat(rows, ignore_index=True)
        out.attrs["pulled_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        out.to_parquet(f)
        return f

    tasks = [(ci, y) for ci in range(len(chunks)) for y in years]
    files = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_retry, job, ci, y): (ci, y) for ci, y in tasks}
        for i, fut in enumerate(as_completed(futs)):
            files.append(fut.result())
            if i % 20 == 0:
                log.info("GEE dynamic %d/%d", i + 1, len(tasks))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["month"] = pd.to_datetime(df["month"])
    return df[[id_col, "month", *bands]].sort_values([id_col, "month"]).reset_index(drop=True)


def extract_static(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    year: int,
    cache_dir: Path,
    chunk: int = 100,
    workers: int = 6,
    ntl_year: int = 2022,
) -> pd.DataFrame:
    """Static covariates (incl. land cover shares) per unit, cached per chunk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf = gdf.sort_values(id_col).reset_index(drop=True)
    chunks = [gdf.iloc[i : i + chunk] for i in range(0, len(gdf), chunk)]
    sb = ["pop_density", "built_frac", "water_occurrence", "ndvi_mean", "ntl_mean", "elevation"]
    lb = list(WORLDCOVER.values())

    def job(ci: int) -> Path:
        f = cache_dir / f"static_{ci:03d}.parquet"
        if f.exists():
            return f
        a = _reduce_with_fallback(
            static_image(year, ntl_year), chunks[ci], id_col, STATIC_SCALE, sb
        )
        b = _reduce_with_fallback(landcover_image(), chunks[ci], id_col, WORLDCOVER_SCALE, lb)
        out = a.merge(b, on=id_col, how="outer")
        out.to_parquet(f)
        return f

    files = []
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(_retry, job, ci) for ci in range(len(chunks))]):
            files.append(fut.result())
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["pop_density_log"] = np.log1p(df["pop_density"])
    return df[[id_col, "pop_density_log", "pop_density", *sb[1:], *lb]]


def _retry(fn, *args, tries: int = 4):
    """Retry an EE job with backoff (EE returns transient 429 and timeout errors)."""
    import time

    for i in range(tries):
        try:
            return fn(*args)
        except ee.EEException as e:
            if i == tries - 1:
                raise
            log.warning("EE error %s, retry %d", str(e)[:120], i + 1)
            time.sleep(15 * 2**i)
