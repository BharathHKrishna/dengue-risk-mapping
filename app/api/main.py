"""FastAPI backend for the dengue risk map.

Serves precomputed predictions from ``app/data`` (written by
``python -m dengue.cli export``) and the static Leaflet frontend.

Endpoints:
- ``GET /risk?region_id=...&month=YYYY-MM``: prediction, interval, drivers, data dates.
- ``GET /map?view=...&month=YYYY-MM``: values for every region (choropleth).
- ``GET /months?view=...``: available months (or years for India states).
- ``GET /meta``: data dates and model notes.
- ``GET /geo/{view}``: simplified boundaries as GeoJSON.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parents[1]
DATA = APP_DIR / "data"
WEB = APP_DIR / "web"

DISCLAIMER = {
    "brazil": "Modeled climate and environment based risk. Validated against reported cases "
    "for Brazil municipalities (out-of-sample for 2023 to 2025). Research prototype, "
    "not an operational forecast.",
    "india_districts": "Modeled relative risk from a model trained in Brazil. NOT validated at "
    "district level: no public district case data were available. Research prototype.",
    "india_states": "State-year comparison of the Brazil-trained model with reported cases "
    "(OpenDengue / NCVBDC). Reported cases are affected by under-reporting.",
}

GEO = {
    "brazil": "brazil_municipalities.geojson",
    "india_districts": "india_districts.geojson",
    "india_states": "india_states.geojson",
}

VAR_LABELS = {
    "t2m_mean": "mean air temperature",
    "t2m_min": "minimum air temperature",
    "t2m_max": "maximum air temperature",
    "rh_mean": "relative humidity",
    "precip_era5": "rainfall (ERA5-Land)",
    "precip_chirps": "rainfall (CHIRPS)",
    "lst_day": "daytime land surface temperature",
    "pop_density_log": "population density",
    "built_frac": "built-up surface share",
    "water_occurrence": "surface water occurrence",
    "ndvi_mean": "vegetation (NDVI)",
    "elevation": "elevation",
    "ntl_mean": "night lights",
    "moy_sin": "season (month of year)",
    "moy_cos": "season (month of year)",
}


def label(feature: str) -> str:
    """Human readable name for a model feature."""
    if "_lag" in feature:
        var, k = feature.rsplit("_lag", 1)
        unit = "month" if k == "1" else "months"
        return f"{VAR_LABELS.get(var, var)}, {k} {unit} earlier"
    if feature.startswith("lc_"):
        return f"land cover share: {feature[3:]}"
    return VAR_LABELS.get(feature, feature)


def inc(v: float | None) -> float | None:
    """Back-transform log1p incidence to cases per 100k (rounded, floored at 0)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(max(0.0, math.expm1(v)), 2)  # incidence cannot be negative


@lru_cache
def tables() -> dict[str, pd.DataFrame]:
    """Load the prediction tables once."""
    br = pd.read_parquet(DATA / "brazil_predictions.parquet")
    br["m"] = br["month"].dt.strftime("%Y-%m")
    ind = pd.read_parquet(DATA / "india_district_predictions.parquet")
    ind["m"] = ind["month"].dt.strftime("%Y-%m")
    st = pd.read_parquet(DATA / "india_state_year.parquet")
    st["region_id"] = "IN-STATE-" + st["state"]
    names = {}
    for f in GEO.values():
        gj = json.loads((DATA / f).read_text(encoding="utf-8"))
        for feat in gj["features"]:
            p = feat["properties"]
            names[p["region_id"]] = p.get("name") or p.get("state")
    return {"brazil": br, "india_districts": ind, "india_states": st, "names": names}


@lru_cache
def meta() -> dict:
    """Data dates written by the export stage."""
    return json.loads((DATA / "meta.json").read_text())


def view_of(region_id: str) -> str:
    """Infer the view from the region id prefix."""
    if region_id.startswith("BR-"):
        return "brazil"
    if region_id.startswith("IN-STATE-"):
        return "india_states"
    if region_id.startswith("IN-D"):
        return "india_districts"
    raise HTTPException(400, "unknown region_id prefix (expected BR-, IN-D or IN-STATE-)")


app = FastAPI(title="Dengue risk map API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/meta")
def get_meta() -> dict:
    """Data dates, interval definition and disclaimers."""
    return {**meta(), "disclaimers": DISCLAIMER}


@app.get("/months")
def months(view: str = Query("brazil", pattern="^(brazil|india_districts|india_states)$")) -> dict:
    """Available months (Brazil, India districts) or years (India states)."""
    t = tables()[view]
    if view == "india_states":
        return {"view": view, "years": sorted(int(y) for y in t["year"].unique())}
    return {"view": view, "months": sorted(t["m"].unique().tolist())}


@app.get("/map")
def map_values(
    view: str = Query("brazil", pattern="^(brazil|india_districts|india_states)$"),
    month: str = Query(..., pattern=r"^\d{4}(-\d{2})?$"),
) -> dict:
    """Per-region values for one month (or year for India states)."""
    t = tables()[view]
    if view == "india_states":
        g = t[t["year"] == int(month[:4])]
        vals = {
            r.region_id: {"pred": round(r.pred_incidence, 2), "obs": round(r.obs_incidence, 2)}
            for r in g.itertuples()
        }
        return {"view": view, "year": int(month[:4]), "values": vals}
    g = t[t["m"] == month]
    if g.empty:
        raise HTTPException(404, f"no data for {month}")
    if view == "brazil":
        vals = {r.region_id: [inc(r.pred), inc(r.obs)] for r in g.itertuples()}
    else:
        vals = {r.region_id: [round(float(r.rel_risk_pct), 1), inc(r.pred)] for r in g.itertuples()}
    return {"view": view, "month": month, "values": vals}


@app.get("/risk")
def risk(
    region_id: str,
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict:
    """Prediction, uncertainty interval, top SHAP drivers and data dates for one region-month."""
    view = view_of(region_id)
    tb = tables()
    t = tb[view]
    base = {
        "region_id": region_id,
        "name": tb["names"].get(region_id),
        "view": view,
        "month": month,
        "data_dates": meta(),
        "disclaimer": DISCLAIMER[view],
    }
    if view == "india_states":
        r = t[(t["region_id"] == region_id) & (t["year"] == int(month[:4]))]
        if r.empty:
            raise HTTPException(404, "no state-year record")
        r = r.iloc[0]
        return {
            **base,
            "year": int(r["year"]),
            "predicted_incidence_per_100k_year": round(float(r["pred_incidence"]), 2),
            "reported_incidence_per_100k_year": round(float(r["obs_incidence"]), 2),
            "reported_cases": int(r["cases"]),
            "note": "Compare ranks across states, not absolute levels.",
        }
    r = t[(t["region_id"] == region_id) & (t["m"] == month)]
    if r.empty:
        raise HTTPException(404, "no prediction for this region and month")
    r = r.iloc[0]
    drivers = [
        {"feature": f, "label": label(f), "shap": s, "direction": "raises" if s > 0 else "lowers"}
        for f, s in json.loads(r["drivers"])
    ]
    out = {
        **base,
        "predicted_incidence_per_100k": inc(float(r["pred"])),
        "interval_per_100k": [inc(float(r["lo"])), inc(float(r["hi"]))],
        "interval_level": meta()["interval"],
        "top_drivers": drivers,
        "drivers_note": "SHAP contributions on the log1p(incidence) scale.",
    }
    if view == "brazil":
        out["observed_incidence_per_100k"] = inc(r["obs"]) if pd.notna(r["obs"]) else None
        out["observed_cases"] = int(r["cases"]) if pd.notna(r["cases"]) else None
        out["prediction_source"] = r["source"]
    else:
        out["relative_risk_percentile"] = round(float(r["rel_risk_pct"]), 1)
        ood = json.loads(r["ood"]) if "ood" in r and isinstance(r["ood"], str) else []
        out["inputs_outside_training_range"] = [label(f) for f in ood]
        if ood:
            out["extrapolation_warning"] = (
                "Some inputs are outside the range seen in Brazil training data, so this "
                "prediction extrapolates and is less reliable."
            )
    return out


@app.get("/geo/{view}")
def geo(view: str) -> FileResponse:
    """Simplified boundaries."""
    if view not in GEO:
        raise HTTPException(404, "unknown view")
    return FileResponse(DATA / GEO[view], media_type="application/geo+json")


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
