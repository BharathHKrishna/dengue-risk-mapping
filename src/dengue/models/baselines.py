"""Seasonal baselines on log incidence."""

from __future__ import annotations

import pandas as pd


def climatology(
    train: pd.DataFrame, target: pd.DataFrame, group: str | None = "geocode"
) -> pd.Series:
    """Average training ``y`` for the same unit and month of year.

    With ``group=None`` this is the pooled climatology (same month of year,
    averaged over all training units), which is the only climatology available
    for a place with no case history. Missing unit-months fall back to the
    pooled value.
    """
    pooled = train.groupby("month_of_year")["y"].mean()
    base = target["month_of_year"].map(pooled)
    if group is None:
        return base.rename("clim")
    local = train.groupby([group, "month_of_year"])["y"].mean().rename("clim").reset_index()
    out = target[[group, "month_of_year"]].merge(local, on=[group, "month_of_year"], how="left")
    out.index = target.index
    return out["clim"].fillna(base).rename("clim")


def last_year(target: pd.DataFrame) -> pd.Series:
    """Same month last year (``y_lag12``). NaN where that month is missing."""
    return target["y_lag12"].rename("last_year")
