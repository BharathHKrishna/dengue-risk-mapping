"""Later pipeline stages: explainability, final model, India transfer, map export."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dengue.experiments import load_table
from dengue.explain.shap_analysis import compute_shap, lag_importance, plots, sanity_checks
from dengue.features.build import feature_columns
from dengue.models.final import build_unlabeled, load_final, predict_final, train_final
from dengue.utils.config import ROOT, path
from dengue.utils.log import get_logger
from dengue.validation.splits import temporal_split
from dengue.validation.transfer import state_year_predictions, transfer_table

log = get_logger(__name__)
APP_DATA = ROOT / "app" / "data"


def run_explain(cfg: dict) -> None:
    """SHAP on temporal test rows for both LightGBM variants, plus biological checks."""
    df = load_table(cfg)
    test = df[temporal_split(df, cfg)["test"]]
    for name, ar in [("lgbm_climate", False), ("lgbm_ar", True)]:
        cols = feature_columns(cfg, ar)
        mf = str(path(cfg, "results", "models", f"{name}_temporal_seed0.txt"))
        Xs, sv = compute_shap(mf, test[cols])
        plots(cfg, Xs, sv, name)
        sanity_checks(Xs, sv).to_csv(path(cfg, "results", f"shap_sanity_{name}.csv"), index=False)
        lag_importance(sv, cfg).to_csv(path(cfg, "results", f"shap_lag_importance_{name}.csv"))
        imp = sv.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
        imp.to_csv(path(cfg, "results", f"shap_importance_{name}.csv"))
        log.info("SHAP done for %s", name)


def run_final(cfg: dict) -> None:
    """Train and save the deployment model."""
    train_final(load_table(cfg), cfg)


def _drivers(contrib: np.ndarray, cols: list[str], k: int = 3) -> list[str]:
    """JSON list of the top ``k`` signed SHAP drivers for each row."""
    idx = np.argsort(-np.abs(contrib), axis=1)[:, :k]
    out = []
    for r, ii in enumerate(idx):
        out.append(json.dumps([[cols[i], round(float(contrib[r, i]), 3)] for i in ii]))
    return out


def _next_month(m: str) -> str:
    return str(pd.Period(m, "M") + 1)


def run_transfer(cfg: dict) -> None:
    """Predict Indian districts with the Brazil model; compare state-year ranks."""
    from dengue.ingest.opendengue import india_national_monthly

    models = load_final(cfg)
    clim = pd.read_parquet(path(cfg, "interim", "india_climate_monthly.parquet"))
    static = pd.read_parquet(path(cfg, "interim", "india_static.parquet"))
    dist = gpd.read_parquet(path(cfg, "interim", "india_districts_simplified.parquet"))
    end = _next_month(cfg["gee"]["climate_end"])
    X = build_unlabeled(
        clim,
        static.drop(columns=["pop_density", "pop", "area_km2"]),
        cfg,
        "district_id",
        cfg["india"]["map_month_start"],
        end,
    )
    pred, contrib = predict_final(models, X)
    dp = pd.concat([X[["district_id", "month"]], pred], axis=1)
    dp["drivers"] = _drivers(contrib, models["cols"])
    dp.to_parquet(path(cfg, "results", "india_district_predictions.parquet"))

    districts = dist[["district_id", "state_current"]].merge(
        static[["district_id", "pop", "area_km2"]]
    )
    annual = state_year_predictions(dp[["district_id", "month", "pred"]], districts)
    st = districts.assign(state=districts["state_current"].replace({"Ladakh": "Jammu & Kashmir"}))
    st = st.groupby("state", as_index=False)[["pop", "area_km2"]].sum()
    st["pop_density_w"] = st["pop"] / st["area_km2"]
    obs = pd.read_parquet(path(cfg, "interim", "india_state_year_cases.parquet"))
    years = list(range(cfg["india"]["label_years"][0], cfg["india"]["label_years"][1] + 1))
    metrics, merged = transfer_table(annual, obs, st[["state", "pop_density_w"]], years)
    metrics.to_csv(path(cfg, "results", "transfer_india_by_year.csv"), index=False)
    merged.to_csv(path(cfg, "results", "transfer_india_state_year.csv"), index=False)
    summ = (
        metrics.groupby("method")
        .agg(
            years=("year", "count"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            min_spearman=("spearman", "min"),
            max_spearman=("spearman", "max"),
            mean_top5_overlap=("top5_overlap", "mean"),
        )
        .reset_index()
    )
    summ.to_csv(path(cfg, "results", "transfer_india_summary.csv"), index=False)
    log.info("transfer summary\n%s", summ.to_string())

    # Seasonality check against reported national monthly cases.
    nat = india_national_monthly(cfg)
    w = dp.merge(districts[["district_id", "pop"]], on="district_id")
    w["exp_cases"] = np.expm1(w["pred"]).clip(lower=0) * w["pop"] / 1e5
    pn = w.groupby("month", as_index=False)["exp_cases"].sum()
    s = nat.merge(pn, on="month")
    rows = []
    for yr, g in s.groupby(s["month"].dt.year):
        if len(g) >= 10:
            rho = spearmanr(g["cases"], g["exp_cases"]).statistic
            rows.append(
                {
                    "year": yr,
                    "months": len(g),
                    "spearman_monthly": float(rho),
                    "obs_peak_month": int(g.loc[g["cases"].idxmax(), "month"].month),
                    "pred_peak_month": int(g.loc[g["exp_cases"].idxmax(), "month"].month),
                }
            )
    pd.DataFrame(rows).to_csv(path(cfg, "results", "transfer_india_seasonality.csv"), index=False)
    s.to_csv(path(cfg, "results", "transfer_india_national_monthly.csv"), index=False)


def india_ood_flags(cfg: dict, cols: list[str], start: str, end: str) -> pd.DataFrame:
    """Per India district-month, the model inputs outside the Brazil training range.

    Tree models extrapolate flat outside their training range, so these
    predictions rest on effects never observed in Brazil. Returns district_id,
    month and ``ood`` (JSON list of feature names, empty list if none).
    """
    br = load_table(cfg)
    br = br[br["year"].between(*cfg["final"]["train_years"])]
    lo, hi = br[cols].min(), br[cols].max()
    clim = pd.read_parquet(path(cfg, "interim", "india_climate_monthly.parquet"))
    static = pd.read_parquet(path(cfg, "interim", "india_static.parquet"))
    X = build_unlabeled(
        clim,
        static.drop(columns=["pop_density", "pop", "area_km2"]),
        cfg,
        "district_id",
        start,
        end,
    )
    out = (X[cols] < lo) | (X[cols] > hi)
    names = np.array(cols)
    X["ood"] = [json.dumps(names[r].tolist()) for r in out.to_numpy()]
    return X[["district_id", "month", "ood"]]


def run_export(cfg: dict) -> None:
    """Write compact prediction tables and boundaries for the web app."""
    APP_DATA.mkdir(parents=True, exist_ok=True)
    models = load_final(cfg)
    end = _next_month(cfg["gee"]["climate_end"])
    start = cfg["final"]["map_start"]

    # Brazil: temporal-test months use out-of-sample temporal predictions;
    # later months use the final model. Observed values attached where available.
    clim = pd.read_parquet(path(cfg, "interim", "brazil_climate_monthly.parquet"))
    static = pd.read_parquet(path(cfg, "interim", "brazil_static.parquet"))
    X = build_unlabeled(clim, static.drop(columns=["pop_density"]), cfg, "geocode", start, end)
    pred, contrib = predict_final(models, X)
    bp = pd.concat([X[["geocode", "month"]], pred], axis=1)
    bp["drivers"] = _drivers(contrib, models["cols"])
    bp["source"] = "final model"
    tt = pd.read_parquet(path(cfg, "results", "temporal_test_predictions.parquet"))
    tt = tt.rename(
        columns={
            "pred_lgbm_climate": "pred_t",
            "lo_lgbm_climate": "lo_t",
            "hi_lgbm_climate": "hi_t",
        }
    )
    bp = bp.merge(
        tt[["geocode", "month", "y", "pred_t", "lo_t", "hi_t"]], on=["geocode", "month"], how="left"
    )
    has_t = bp["pred_t"].notna()
    bp.loc[has_t, "pred"] = bp.loc[has_t, "pred_t"]
    bp.loc[has_t, "lo"] = np.minimum(bp.loc[has_t, "lo_t"], bp.loc[has_t, "pred"])
    bp.loc[has_t, "hi"] = np.maximum(bp.loc[has_t, "hi_t"], bp.loc[has_t, "pred"])
    bp.loc[has_t, "source"] = "temporal hold-out model (out of sample)"
    obs = load_table(cfg)[["geocode", "month", "y", "cases", "pop"]]
    bp = bp.drop(columns=["y", "pred_t", "lo_t", "hi_t"]).merge(
        obs, on=["geocode", "month"], how="left"
    )
    bp = bp.rename(columns={"geocode": "region_id", "y": "obs"})
    bp["region_id"] = "BR-" + bp["region_id"].astype(str)
    _round(bp).to_parquet(APP_DATA / "brazil_predictions.parquet", index=False)

    # India districts: relative risk = percentile of predicted incidence among districts that month.
    dp = pd.read_parquet(path(cfg, "results", "india_district_predictions.parquet"))
    dp = dp[dp["month"] >= start].copy()
    dp["rel_risk_pct"] = dp.groupby("month")["pred"].rank(pct=True) * 100
    dp = dp.merge(
        india_ood_flags(cfg, models["cols"], start, end), on=["district_id", "month"], how="left"
    )
    dp = dp.rename(columns={"district_id": "region_id"})
    _round(dp).to_parquet(APP_DATA / "india_district_predictions.parquet", index=False)

    # India states: predicted vs reported per year.
    sy = pd.read_csv(path(cfg, "results", "transfer_india_state_year.csv"))
    sy[["state", "year", "pred_incidence", "obs_incidence", "cases", "pop"]].to_parquet(
        APP_DATA / "india_state_year.parquet", index=False
    )

    _write_geo(cfg)
    # Static covariates and the frozen model, committed so the monthly job can run in CI.
    bs = static.drop(columns=["pop_density"]).assign(
        geocode_id="BR-" + static["geocode"].astype(str)
    )
    bs.drop(columns=["geocode"]).to_parquet(APP_DATA / "brazil_static.parquet", index=False)
    ist = pd.read_parquet(path(cfg, "interim", "india_static.parquet"))
    ist.drop(columns=["pop_density", "pop", "area_km2"]).to_parquet(
        APP_DATA / "india_static.parquet", index=False
    )
    from dengue.update import copy_models_to_app

    copy_models_to_app(cfg)
    meta = {
        "climate_data_through": cfg["gee"]["climate_end"],
        "latest_prediction_month": end,
        "brazil_cases_through": str(obs["month"].max().date())[:7],
        "india_cases_through": int(sy["year"].max()),
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "interval": f"{round((1 - cfg['model']['interval_alpha']) * 100)}% conformal interval",
    }
    (APP_DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("export done: %s", meta)


def _round(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].astype("float32").round(3)
    return df


def _write_geo(cfg: dict) -> None:
    """Simplified GeoJSON boundaries for the map (coordinates rounded to about 100 m)."""
    br = gpd.read_parquet(path(cfg, "interim", "brazil_municipalities.parquet"))
    br["geometry"] = br.geometry.simplify(0.02, preserve_topology=True)
    br = br.assign(region_id="BR-" + br["geocode"].astype(str))[
        ["region_id", "name", "uf", "region", "geometry"]
    ]
    br.to_file(APP_DATA / "brazil_municipalities.geojson", driver="GeoJSON", COORDINATE_PRECISION=3)
    d = gpd.read_parquet(path(cfg, "interim", "india_districts_simplified.parquet"))
    d["geometry"] = d.geometry.simplify(0.02, preserve_topology=True)
    d = d.rename(columns={"district_id": "region_id", "district": "name", "state_current": "state"})
    d[["region_id", "name", "state", "geometry"]].to_file(
        APP_DATA / "india_districts.geojson", driver="GeoJSON", COORDINATE_PRECISION=3
    )
    s = gpd.read_parquet(path(cfg, "interim", "india_states.parquet"))
    s["geometry"] = s.geometry.simplify(0.02, preserve_topology=True)
    s = s.assign(region_id="IN-STATE-" + s["state"])
    s[["region_id", "state", "geometry"]].to_file(
        APP_DATA / "india_states.geojson", driver="GeoJSON", COORDINATE_PRECISION=3
    )


def run_diag_no_elevation(cfg: dict) -> None:
    """Post-hoc diagnostic: drop elevation, which SHAP showed to drive Himalayan predictions.

    Elevation is a regional proxy in Brazil (high-burden plateau) and Indian hill
    states lie outside Brazil's elevation range. This trains the climate-only
    model without elevation (3 seeds, point models), then reports both the
    Brazil temporal skill and the India state rank test. It is a diagnostic
    chosen after seeing the India results, so it is reported next to, never
    instead of, the pre-registered model.
    """
    from dengue.models.baselines import climatology
    from dengue.models.gbm import fit_point
    from dengue.utils.seed import set_seed
    from dengue.validation.metrics import regression_metrics

    df = load_table(cfg)
    cols = [c for c in feature_columns(cfg, False) if c != "elevation"]
    tm = temporal_split(df, cfg)
    tr, te = df[tm["train"]], df[tm["test"]]
    clim = climatology(tr, te).to_numpy()
    preds = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        preds.append(fit_point(tr[cols], tr["y"], cfg, seed).predict(te[cols]))
    tmet = regression_metrics(te["y"].to_numpy(), np.mean(preds, axis=0), base=clim)
    pd.DataFrame([{"model": "lgbm_climate without elevation (3-seed mean)", **tmet}]).to_csv(
        path(cfg, "results", "diag_noelev_temporal.csv"), index=False
    )
    fc = cfg["final"]
    ftr = df[df["year"].between(*fc["train_years"])]
    models = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        models.append(fit_point(ftr[cols], ftr["y"], cfg, seed))
    clim_in = pd.read_parquet(path(cfg, "interim", "india_climate_monthly.parquet"))
    static = pd.read_parquet(path(cfg, "interim", "india_static.parquet"))
    dist = gpd.read_parquet(path(cfg, "interim", "india_districts_simplified.parquet"))
    end = _next_month(cfg["gee"]["climate_end"])
    X = build_unlabeled(
        clim_in,
        static.drop(columns=["pop_density", "pop", "area_km2"]),
        cfg,
        "district_id",
        cfg["india"]["map_month_start"],
        end,
    )
    X["pred"] = np.mean([m.predict(X[cols]) for m in models], axis=0)
    districts = dist[["district_id", "state_current"]].merge(
        static[["district_id", "pop", "area_km2"]]
    )
    annual = state_year_predictions(X[["district_id", "month", "pred"]], districts)
    st = districts.assign(state=districts["state_current"].replace({"Ladakh": "Jammu & Kashmir"}))
    st = st.groupby("state", as_index=False)[["pop", "area_km2"]].sum()
    st["pop_density_w"] = st["pop"] / st["area_km2"]
    obs = pd.read_parquet(path(cfg, "interim", "india_state_year_cases.parquet"))
    years = list(range(cfg["india"]["label_years"][0], cfg["india"]["label_years"][1] + 1))
    metrics, _ = transfer_table(annual, obs, st[["state", "pop_density_w"]], years)
    metrics = metrics[metrics["method"].str.startswith("model")]
    metrics = metrics.assign(method="model without elevation (diagnostic)")
    metrics.to_csv(path(cfg, "results", "diag_noelev_transfer_by_year.csv"), index=False)
    log.info(
        "no-elevation diagnostic: temporal %s, India mean Spearman %.3f",
        {k: round(v, 3) for k, v in tmet.items()},
        metrics["spearman"].mean(),
    )
