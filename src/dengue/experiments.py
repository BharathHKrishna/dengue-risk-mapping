"""Brazil experiments: temporal validation, spatial blocked CV, intervals, alerts.

Every run is logged to MLflow (local ``mlruns/``) with the config, git hash
and metrics. Result tables are written to ``results/`` as CSV and these files
are the only source of the numbers quoted in the README and report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from dengue.features.build import add_high_label, feature_columns
from dengue.models.baselines import climatology, last_year
from dengue.models.gbm import conformal_margin, fit_point, fit_quantiles
from dengue.utils.config import path
from dengue.utils.log import get_logger, git_hash
from dengue.utils.seed import set_seed
from dengue.validation.metrics import (
    alert_metrics,
    fast_cluster_bootstrap,
    interval_coverage,
    spearman,
)
from dengue.validation.splits import spatial_folds, temporal_split

log = get_logger(__name__)

MODELS = {"lgbm_climate": False, "lgbm_ar": True}  # name -> uses autoregressive features


def load_table(cfg: dict) -> pd.DataFrame:
    """Processed Brazil municipality-month table."""
    df = pd.read_parquet(path(cfg, "processed", "brazil_monthly.parquet"))
    return df[df["pop"] > 0].reset_index(drop=True)


def _start_run(cfg: dict, name: str) -> None:
    store = Path(cfg["paths"]["mlruns"])
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (store / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name("dengue-brazil") is None:
        mlflow.create_experiment("dengue-brazil", artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment("dengue-brazil")
    mlflow.start_run(run_name=name)
    mlflow.set_tags({"git_hash": git_hash(), "config": cfg["_config_path"]})
    mlflow.log_params(
        {
            "lgbm": json.dumps(cfg["model"]["lightgbm"]),
            "split": json.dumps(cfg["split"]),
            "lags": json.dumps(cfg["features"]["lags"]),
            "seeds": json.dumps(cfg["seeds"]),
        }
    )


def _score_block(
    name: str, test: pd.DataFrame, pred: np.ndarray, base: np.ndarray, cfg: dict, seed: int = 0
) -> dict:
    """Headline regression metrics with cluster-bootstrap CIs, plus Spearman and alert metrics."""
    ok = ~np.isnan(pred) & ~np.isnan(base)
    t = test[ok]
    ci = fast_cluster_bootstrap(
        t["geocode"].to_numpy(),
        t["y"].to_numpy(),
        pred[ok],
        base[ok],
        n_boot=cfg["eval"]["n_bootstrap"],
        seed=seed,
    )
    row = {"model": name, "n": int(ok.sum())}
    for m in ci.index:
        row[m] = ci.loc[m, "point"]
        row[f"{m}_lo"] = ci.loc[m, "ci_low"]
        row[f"{m}_hi"] = ci.loc[m, "ci_high"]
    row["spearman"] = spearman(t["y"].to_numpy(), pred[ok])
    a = t["high"].notna().to_numpy()
    if a.any():
        score = pred[ok][a] - t["high_thr"].to_numpy()[a]
        am = alert_metrics(t["high"].to_numpy()[a].astype(int), score, 0.0)
        row.update({f"alert_{k}": v for k, v in am.items()})
    return row


def _fit_predict(
    df: pd.DataFrame, masks: dict, cfg: dict, autoregressive: bool, quantiles: bool
) -> dict:
    """Fit 3 seeds; return seed predictions on test and (optionally) conformal intervals."""
    cols = feature_columns(cfg, autoregressive)
    tr, ca, te = df[masks["train"]], df[masks["calib"]], df[masks["test"]]
    preds, intervals = [], []
    for seed in cfg["seeds"]:
        set_seed(seed)
        t0 = time.time()
        m = fit_point(tr[cols], tr["y"], cfg, seed)
        preds.append(m.predict(te[cols]))
        if seed == cfg["seeds"][0]:
            point_model = m
        if quantiles:
            qm = fit_quantiles(tr[cols], tr["y"], cfg, seed)
            qs = sorted(qm)
            lo_c, hi_c = qm[qs[0]].predict(ca[cols]), qm[qs[-1]].predict(ca[cols])
            margin = conformal_margin(
                ca["y"].to_numpy(), lo_c, hi_c, cfg["model"]["interval_alpha"]
            )
            lo_t, hi_t = qm[qs[0]].predict(te[cols]), qm[qs[-1]].predict(te[cols])
            intervals.append(
                {
                    "seed": seed,
                    "margin": margin,
                    "raw": interval_coverage(te["y"].to_numpy(), lo_t, hi_t),
                    "conformal": interval_coverage(
                        te["y"].to_numpy(), lo_t - margin, hi_t + margin
                    ),
                    "lo": lo_t - margin,
                    "hi": hi_t + margin,
                }
            )
        log.info("seed %d fitted in %.0fs", seed, time.time() - t0)
    return {"preds": np.vstack(preds), "intervals": intervals, "model": point_model, "cols": cols}


def run_temporal(cfg: dict) -> pd.DataFrame:
    """Temporal hold-out: train 2015-2021, calibrate 2022, test 2023-2025."""
    df = load_table(cfg)
    masks = temporal_split(df, cfg)
    df = add_high_label(df, masks["train"], cfg["target"]["high_quantile"])
    test = df[masks["test"]]
    train = df[masks["train"]]
    clim = climatology(train, test).to_numpy()
    rows = [
        _score_block("climatology", test, clim, clim, cfg),
        _score_block("last_year", test, last_year(test).to_numpy(), clim, cfg),
    ]
    seed_rows, iv_rows, out_preds = [], [], test[["geocode", "month", "y"]].copy()
    out_preds["clim"] = clim
    _start_run(cfg, "temporal")
    for name, ar in MODELS.items():
        res = _fit_predict(df, masks, cfg, ar, quantiles=True)
        ens = res["preds"].mean(axis=0)
        rows.append(_score_block(f"{name} (3-seed mean)", test, ens, clim, cfg))
        for s, p in zip(cfg["seeds"], res["preds"], strict=True):
            r = _score_block(name, test, p, clim, cfg, seed=s)
            seed_rows.append(
                {
                    "model": name,
                    "seed": s,
                    **{
                        k: r[k]
                        for k in ("mae", "rmse", "skill_vs_clim", "spearman", "alert_auc_pr")
                    },
                }
            )
        for iv in res["intervals"]:
            iv_rows.append(
                {
                    "model": name,
                    "seed": iv["seed"],
                    "nominal": 1 - cfg["model"]["interval_alpha"],
                    "raw_coverage": iv["raw"]["coverage"],
                    "raw_width": iv["raw"]["mean_width"],
                    "conformal_margin": iv["margin"],
                    "conformal_coverage": iv["conformal"]["coverage"],
                    "conformal_width": iv["conformal"]["mean_width"],
                }
            )
        out_preds[f"pred_{name}"] = ens
        out_preds[f"lo_{name}"] = np.mean([iv["lo"] for iv in res["intervals"]], axis=0)
        out_preds[f"hi_{name}"] = np.mean([iv["hi"] for iv in res["intervals"]], axis=0)
        res["model"].booster_.save_model(
            str(path(cfg, "results", "models", f"{name}_temporal_seed0.txt"))
        )
    table = pd.DataFrame(rows)
    seeds = pd.DataFrame(seed_rows)
    ivs = pd.DataFrame(iv_rows)
    table.to_csv(path(cfg, "results", "temporal_metrics.csv"), index=False)
    seeds.to_csv(path(cfg, "results", "temporal_seed_metrics.csv"), index=False)
    ivs.to_csv(path(cfg, "results", "temporal_intervals.csv"), index=False)
    out_preds.to_parquet(path(cfg, "results", "temporal_test_predictions.parquet"))
    # Year-by-year breakdown, because 2024 was an exceptional epidemic year in Brazil.
    by_year = []
    for yr, g in out_preds.assign(year=out_preds["month"].dt.year).groupby("year"):
        for c in ["clim", *[f"pred_{n}" for n in MODELS]]:
            e = g["y"] - g[c]
            by_year.append(
                {
                    "year": yr,
                    "model": c,
                    "mae": e.abs().mean(),
                    "rmse": np.sqrt((e**2).mean()),
                    "skill_vs_clim": 1 - (e**2).mean() / ((g["y"] - g["clim"]) ** 2).mean(),
                }
            )
    pd.DataFrame(by_year).to_csv(path(cfg, "results", "temporal_by_year.csv"), index=False)
    for _, r in table.iterrows():
        key = r["model"].split(" ")[0]
        mlflow.log_metrics(
            {f"{key}_{m}": float(r[m]) for m in ("mae", "rmse", "skill_vs_clim", "spearman")}
        )
    mlflow.log_artifact(str(path(cfg, "results", "temporal_metrics.csv")))
    mlflow.end_run()
    return table


def run_spatial(cfg: dict) -> pd.DataFrame:
    """Leave-one-macro-region-out CV, blocked in space and time."""
    df = load_table(cfg)
    tm = temporal_split(df, cfg)
    # Thresholds for the alert label come from each unit's own training years.
    # For a held-out region this uses its past labels only to define "high",
    # never as model input.
    df = add_high_label(df, tm["train"], cfg["target"]["high_quantile"])
    rows, iv_rows = [], []
    _start_run(cfg, "spatial")
    for region, masks in spatial_folds(df, cfg):
        test, train = df[masks["test"]], df[masks["train"]]
        pooled = climatology(train, test, group=None).to_numpy()
        # Reference only: the held-out places' own history from training years.
        local = climatology(df[tm["train"] & (df["region"] == region)], test).to_numpy()
        rows.append(
            {"region": region, **_score_block("pooled_climatology", test, pooled, pooled, cfg)}
        )
        rows.append(
            {
                "region": region,
                **_score_block("local_climatology (uses own history)", test, local, pooled, cfg),
            }
        )
        for name, ar in MODELS.items():
            if ar:
                continue  # AR features need the held-out places' past cases; see run_spatial_ar.
            res = _fit_predict(df, masks, cfg, ar, quantiles=True)
            ens = res["preds"].mean(axis=0)
            rows.append(
                {"region": region, **_score_block(f"{name} (3-seed mean)", test, ens, pooled, cfg)}
            )
            r_local = _score_block(name, test, ens, local, cfg)
            rows[-1]["skill_vs_local_clim"] = r_local["skill_vs_clim"]
            for iv in res["intervals"]:
                iv_rows.append(
                    {
                        "region": region,
                        "model": name,
                        "seed": iv["seed"],
                        "conformal_coverage": iv["conformal"]["coverage"],
                        "raw_coverage": iv["raw"]["coverage"],
                        "conformal_width": iv["conformal"]["mean_width"],
                    }
                )
        log.info("spatial fold %s done", region)
    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "results", "spatial_metrics.csv"), index=False)
    pd.DataFrame(iv_rows).to_csv(path(cfg, "results", "spatial_intervals.csv"), index=False)
    mlflow.log_artifact(str(path(cfg, "results", "spatial_metrics.csv")))
    mlflow.end_run()
    return table


def run_ablation(cfg: dict) -> pd.DataFrame:
    """Does climate add skill beyond each place's own seasonal history?

    Compares, on the temporal test years, three LightGBM variants (3 seeds,
    point models only) that never see recent cases:
    - ``climatology_only``: the local climatology value plus month of year.
    - ``climatology_plus_static``: adds static covariates.
    - ``climatology_plus_climate``: adds the lagged climate features as well.
    The climatology feature is each municipality's mean ``y`` for that month of
    year over training years only, so it carries no test-period information.
    """
    df = load_table(cfg)
    masks = temporal_split(df, cfg)
    df = add_high_label(df, masks["train"], cfg["target"]["high_quantile"])
    df["clim_local"] = climatology(df[masks["train"]], df).to_numpy()
    train, test = df[masks["train"]], df[masks["test"]]
    base = test["clim_local"].to_numpy()
    climate_cols = [c for c in feature_columns(cfg, False) if "_lag" in c]
    static_cols = list(cfg["features"]["static_vars"])
    variants = {
        "climatology_only": ["clim_local", "moy_sin", "moy_cos"],
        "climatology_plus_static": ["clim_local", "moy_sin", "moy_cos", *static_cols],
        "climatology_plus_climate": [
            "clim_local",
            "moy_sin",
            "moy_cos",
            *static_cols,
            *climate_cols,
        ],
    }
    rows = [_score_block("climatology (no model)", test, base, base, cfg)]
    for name, cols in variants.items():
        preds = []
        for seed in cfg["seeds"]:
            set_seed(seed)
            preds.append(fit_point(train[cols], train["y"], cfg, seed).predict(test[cols]))
        rows.append(_score_block(f"{name} (3-seed mean)", test, np.mean(preds, axis=0), base, cfg))
        log.info("ablation %s done", name)
    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "results", "ablation_climate_value.csv"), index=False)
    return table
