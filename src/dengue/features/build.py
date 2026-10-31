"""Feature engineering: monthly labels, lagged climate, seasonality, static covariates.

Leakage rule: every feature attached to month ``t`` is computed only from data of
months ``<= t - 1``. Lags are built by joining on an explicit month offset, not
by row shifting, so gaps in the series can never pull in the wrong month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add incidence per 100k and ``y = log1p(incidence)``.

    Calendar months hold 4 or 5 epi weeks, so when ``n_weeks`` is present the
    count is scaled to a standard month of 52.18 / 12 = 4.345 weeks.
    """
    out = df.copy()
    cases = out["cases"]
    if "n_weeks" in out:
        cases = cases / out["n_weeks"] * (52.18 / 12)
    out["incidence"] = cases / out["pop"] * 1e5
    out["y"] = np.log1p(out["incidence"])
    return out


def add_lags(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int],
    group: str = "geocode",
    time: str = "month",
    suffix: str = "lag",
) -> pd.DataFrame:
    """Attach ``{col}_{suffix}{k}`` = value of ``col`` at month ``t - k`` for the same unit.

    ``time`` must hold month-start Timestamps. Missing months give NaN, never
    a neighbouring month.
    """
    if any(k < 1 for k in lags):
        raise ValueError("lags must be >= 1 to avoid using same-month information")
    out = df.copy()
    src = df[[group, time, *cols]]
    for k in lags:
        shifted = src.copy()
        shifted[time] = shifted[time] + pd.DateOffset(months=k)
        shifted = shifted.rename(columns={c: f"{c}_{suffix}{k}" for c in cols})
        out = out.merge(shifted, on=[group, time], how="left")
    return out


def add_seasonality(df: pd.DataFrame, time: str = "month") -> pd.DataFrame:
    """Cyclic month-of-year encoding."""
    out = df.copy()
    moy = out[time].dt.month
    out["month_of_year"] = moy
    out["moy_sin"] = np.sin(2 * np.pi * moy / 12)
    out["moy_cos"] = np.cos(2 * np.pi * moy / 12)
    return out


def add_high_label(
    df: pd.DataFrame, train_mask: pd.Series, q: float = 0.75, group: str = "geocode"
) -> pd.DataFrame:
    """Binary high-incidence label using each unit's own ``q`` quantile of ``y``.

    Thresholds are computed on training rows only (``train_mask``) and then
    applied to all rows, so test months never influence the threshold. A
    month is "high" when ``y`` is strictly above the threshold and at least
    one case was reported.
    """
    out = df.copy()
    thr = out.loc[train_mask].groupby(group)["y"].quantile(q).rename("high_thr")
    out = out.drop(columns=["high_thr"], errors="ignore").merge(
        thr, left_on=group, right_index=True, how="left"
    )
    out["high"] = ((out["y"] > out["high_thr"]) & (out["cases"] > 0)).astype(int)
    out.loc[out["high_thr"].isna(), "high"] = np.nan
    return out


def feature_columns(cfg: dict, autoregressive: bool) -> list[str]:
    """Model input columns. ``autoregressive`` adds lagged log incidence."""
    lags = cfg["features"]["lags"]
    cols = [f"{v}_lag{k}" for v in cfg["features"]["climate_vars"] for k in lags]
    cols += ["moy_sin", "moy_cos"]
    cols += list(cfg["features"]["static_vars"])
    if autoregressive:
        cols += [f"y_lag{k}" for k in cfg["features"]["ar_lags"]]
    return cols


def build_table(
    cases_monthly: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    static: pd.DataFrame,
    cfg: dict,
    group: str = "geocode",
) -> pd.DataFrame:
    """Join labels, lagged climate, seasonality and static covariates into one table.

    ``cases_monthly``: group, month, cases, pop. ``climate_monthly``: group,
    month, climate vars (may start before the label window, for lags).
    ``static``: group plus static vars. Rows are restricted to the label window.
    """
    climate_vars = cfg["features"]["climate_vars"]
    lab = add_target(cases_monthly)
    lab = add_lags(lab, ["y"], cfg["features"]["ar_lags"], group=group)
    clim = add_lags(climate_monthly, climate_vars, cfg["features"]["lags"], group=group)
    lag_cols = [c for c in clim.columns if "_lag" in c]
    df = lab.merge(clim[[group, "month", *lag_cols]], on=[group, "month"], how="left")
    df = df.merge(static, on=group, how="left")
    df = add_seasonality(df)
    start = pd.Timestamp(cfg["brazil"]["label_start"])
    end = pd.Timestamp(cfg["brazil"]["label_end"])
    df = df[(df["month"] >= start) & (df["month"] <= end)].reset_index(drop=True)
    df["year"] = df["month"].dt.year
    return df
