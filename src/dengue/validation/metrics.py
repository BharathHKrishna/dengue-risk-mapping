"""Metrics and cluster bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination against the test-set mean."""
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def skill(y: np.ndarray, p: np.ndarray, base: np.ndarray) -> float:
    """MSE skill score versus a baseline: 1 - MSE(model) / MSE(baseline).

    Positive means the model beats the baseline. This is the "R2 against the
    climatology baseline" in the brief.
    """
    return float(1 - np.mean((y - p) ** 2) / np.mean((y - base) ** 2))


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    """Spearman rank correlation."""
    return float(spearmanr(y, p).statistic)


def regression_metrics(y: np.ndarray, p: np.ndarray, base: np.ndarray | None = None) -> dict:
    """MAE, RMSE, R2, Spearman, and skill versus ``base`` if given."""
    out = {"mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p), "spearman": spearman(y, p)}
    if base is not None:
        out["skill_vs_clim"] = skill(y, p, base)
    return out


def alert_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1 at ``threshold`` and AUC-PR (average precision) of ``score``."""
    pred = (score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, score)),
        "prevalence": float(np.mean(y_true)),
    }


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Empirical coverage and mean width of prediction intervals."""
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict],
    cluster: str = "geocode",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap CIs by resampling whole clusters (municipalities) with replacement.

    Rows from one municipality are strongly correlated over time, so
    resampling rows would give intervals that are too narrow. Returns a frame
    with point estimate, 2.5 and 97.5 percentiles per metric.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster).indices
    keys = np.array(list(groups))
    point = metric_fn(df)
    samples = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        samples.append(metric_fn(df.iloc[idx]))
    s = pd.DataFrame(samples)
    return pd.DataFrame(
        {"point": pd.Series(point), "ci_low": s.quantile(0.025), "ci_high": s.quantile(0.975)}
    )
