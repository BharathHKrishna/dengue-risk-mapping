"""LightGBM point and quantile models, plus split conformal calibration."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def _params(cfg: dict, seed: int, **override) -> dict:
    p = dict(cfg["model"]["lightgbm"])
    p.update(seed=seed, deterministic=True, force_row_wise=True, n_jobs=-1)
    p.update(override)
    return p


def fit_point(X: pd.DataFrame, y: pd.Series, cfg: dict, seed: int) -> lgb.LGBMRegressor:
    """Fit the main LightGBM regressor on log incidence."""
    model = lgb.LGBMRegressor(**_params(cfg, seed))
    model.fit(X, y)
    return model


def fit_quantiles(
    X: pd.DataFrame, y: pd.Series, cfg: dict, seed: int
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one LightGBM model per quantile in ``cfg.model.quantiles``."""
    models = {}
    for q in cfg["model"]["quantiles"]:
        m = lgb.LGBMRegressor(**_params(cfg, seed, objective="quantile", alpha=q))
        m.fit(X, y)
        models[q] = m
    return models


def conformal_margin(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Conformalized quantile regression (Romano et al. 2019) margin from a calibration set.

    Scores are ``max(lo - y, y - hi)``. The returned margin ``m`` widens (or
    narrows, if negative) the interval to ``[lo - m, hi + m]`` so that it covers
    at least ``1 - alpha`` of exchangeable new points.
    """
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))
