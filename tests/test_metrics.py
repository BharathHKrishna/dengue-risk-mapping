"""Tests for metrics, conformal margin and baselines."""

import numpy as np
import pandas as pd

from dengue.models.baselines import climatology
from dengue.models.gbm import conformal_margin
from dengue.validation.metrics import (
    alert_metrics,
    cluster_bootstrap,
    interval_coverage,
    regression_metrics,
    skill,
)


def test_regression_metrics_perfect_prediction():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = regression_metrics(y, y, base=np.zeros(4))
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1 and m["skill_vs_clim"] == 1


def test_skill_zero_when_equal_to_baseline():
    y = np.array([1.0, 2.0, 3.0])
    b = np.array([1.5, 1.5, 1.5])
    assert skill(y, b, b) == 0


def test_alert_metrics_known_case():
    m = alert_metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1]), 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_conformal_margin_gives_nominal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # deliberately too narrow
    m = conformal_margin(y[:10000], lo[:10000], hi[:10000], alpha=0.1)
    cov = interval_coverage(y[10000:], lo[10000:] - m, hi[10000:] + m)["coverage"]
    assert abs(cov - 0.9) < 0.02


def test_cluster_bootstrap_brackets_point():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"geocode": np.repeat(np.arange(50), 10), "e": rng.normal(size=500)})
    ci = cluster_bootstrap(df, lambda d: {"m": d["e"].mean()}, n_boot=200)
    assert ci.loc["m", "ci_low"] <= ci.loc["m", "point"] <= ci.loc["m", "ci_high"]


def test_climatology_local_and_pooled():
    train = pd.DataFrame({"geocode": [1, 1, 2], "month_of_year": [1, 1, 1], "y": [1.0, 3.0, 5.0]})
    target = pd.DataFrame({"geocode": [1, 3], "month_of_year": [1, 1]})
    loc = climatology(train, target)
    assert loc.tolist() == [2.0, 3.0]  # unit 3 unseen, falls back to pooled mean
    assert climatology(train, target, group=None).tolist() == [3.0, 3.0]
