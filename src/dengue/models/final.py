"""Final (deployment) models and unlabeled feature building for prediction.

The deployment model is the climate-only LightGBM (no past cases), because
real-time case data are not available for India and are delayed in Brazil.
It is trained on Brazil label years ``final_train_years`` and its intervals
are conformalized on ``final_calib_years``.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from dengue.features.build import add_lags, add_seasonality, feature_columns
from dengue.models.gbm import conformal_margin, fit_point, fit_quantiles
from dengue.utils.config import path
from dengue.utils.log import get_logger
from dengue.utils.seed import set_seed

log = get_logger(__name__)


def build_unlabeled(
    climate: pd.DataFrame, static: pd.DataFrame, cfg: dict, group: str, start: str, end: str
) -> pd.DataFrame:
    """Features for every unit-month in ``[start, end]`` without labels.

    A month ``t`` can be predicted once climate for ``t - 1`` exists, so the
    last predictable month is one month after the last climate month.
    """
    vars_ = cfg["features"]["climate_vars"]
    months = pd.date_range(start, end, freq="MS")
    grid = pd.MultiIndex.from_product([climate[group].unique(), months], names=[group, "month"])
    grid = grid.to_frame(index=False)
    full = pd.concat([climate, grid[~grid["month"].isin(climate["month"])]], ignore_index=True)
    lagged = add_lags(full, vars_, cfg["features"]["lags"], group=group)
    lag_cols = [c for c in lagged.columns if "_lag" in c]
    df = grid.merge(lagged[[group, "month", *lag_cols]], on=[group, "month"], how="left")
    df = df.merge(static, on=group, how="left")
    return add_seasonality(df)


def train_final(df: pd.DataFrame, cfg: dict) -> dict:
    """Train 3-seed point models and quantile models; return models and conformal margin."""
    fc = cfg["final"]
    cols = feature_columns(cfg, autoregressive=False)
    tr = df[df["year"].between(*fc["train_years"])]
    ca = df[df["year"].between(*fc["calib_years"])]
    points = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        points.append(fit_point(tr[cols], tr["y"], cfg, seed))
    qm = fit_quantiles(tr[cols], tr["y"], cfg, cfg["seeds"][0])
    qs = sorted(qm)
    margin = conformal_margin(
        ca["y"].to_numpy(),
        qm[qs[0]].predict(ca[cols]),
        qm[qs[-1]].predict(ca[cols]),
        cfg["model"]["interval_alpha"],
    )
    for i, m in enumerate(points):
        m.booster_.save_model(str(path(cfg, "results", "models", f"final_point_seed{i}.txt")))
    qm[qs[0]].booster_.save_model(str(path(cfg, "results", "models", "final_q_lo.txt")))
    qm[qs[-1]].booster_.save_model(str(path(cfg, "results", "models", "final_q_hi.txt")))
    (path(cfg, "results", "models", "final_margin.txt")).write_text(f"{margin:.6f}\n")
    log.info("final model trained, conformal margin %.4f", margin)
    return {"points": points, "lo": qm[qs[0]], "hi": qm[qs[-1]], "margin": margin, "cols": cols}


def load_final(cfg: dict) -> dict:
    """Load saved final boosters."""
    d = path(cfg, "results", "models", "x").parent
    points = [
        lgb.Booster(model_file=str(d / f"final_point_seed{i}.txt"))
        for i in range(len(cfg["seeds"]))
    ]
    return {
        "points": points,
        "lo": lgb.Booster(model_file=str(d / "final_q_lo.txt")),
        "hi": lgb.Booster(model_file=str(d / "final_q_hi.txt")),
        "margin": float((d / "final_margin.txt").read_text()),
        "cols": feature_columns(cfg, autoregressive=False),
    }


def predict_final(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Point (3-seed mean), conformal interval, and per-row SHAP contributions (seed 0)."""
    cols = models["cols"]

    def pred(m, x):
        return m.predict(x) if isinstance(m, lgb.Booster) else m.booster_.predict(x)

    pts = np.mean([pred(m, X[cols]) for m in models["points"]], axis=0)
    lo = pred(models["lo"], X[cols]) - models["margin"]
    hi = pred(models["hi"], X[cols]) + models["margin"]
    b0 = models["points"][0]
    b0 = b0 if isinstance(b0, lgb.Booster) else b0.booster_
    contrib = b0.predict(X[cols], pred_contrib=True)[:, :-1]  # last column is the bias
    out = pd.DataFrame(
        {"pred": pts, "lo": np.minimum(lo, pts), "hi": np.maximum(hi, pts)}, index=X.index
    )
    return out, contrib


def ood_lists(X: pd.DataFrame, cols: list[str], ranges: dict[str, list[float]]) -> list[str]:
    """For each row, a JSON list of features outside the training ``[min, max]`` range."""
    import json

    lo = pd.Series({c: ranges[c][0] for c in cols})
    hi = pd.Series({c: ranges[c][1] for c in cols})
    out = (X[cols] < lo) | (X[cols] > hi)
    names = np.array(cols)
    return [json.dumps(names[r].tolist()) for r in out.to_numpy()]
