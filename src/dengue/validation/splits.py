"""Temporal and spatial split logic with explicit leakage checks."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def year_mask(df: pd.DataFrame, years: list[int]) -> pd.Series:
    """Rows whose ``year`` lies in the inclusive range ``[years[0], years[1]]``."""
    return df["year"].between(years[0], years[1])


def temporal_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Masks for train, calibration and test periods. Never shuffles time."""
    s = cfg["split"]
    masks = {
        "train": year_mask(df, s["train_years"]),
        "calib": year_mask(df, s["calib_years"]),
        "test": year_mask(df, s["test_years"]),
    }
    assert_temporal_order(df, masks)
    return masks


def assert_temporal_order(df: pd.DataFrame, masks: dict[str, pd.Series]) -> None:
    """Raise if any train month is not strictly before every calib and test month."""
    t_train = df.loc[masks["train"], "month"].max()
    for k in ("calib", "test"):
        if masks[k].any() and df.loc[masks[k], "month"].min() <= t_train:
            raise AssertionError(f"{k} period overlaps or precedes training period")
    if masks["calib"].any() and masks["test"].any():
        if df.loc[masks["test"], "month"].min() <= df.loc[masks["calib"], "month"].max():
            raise AssertionError("test period overlaps calibration period")


def spatial_folds(
    df: pd.DataFrame, cfg: dict, group_col: str = "region"
) -> Iterator[tuple[str, dict[str, pd.Series]]]:
    """Leave-one-group-out folds that are blocked in both space and time.

    For held-out group ``g``: train on other groups in training years,
    calibrate on other groups in calibration years, test on ``g`` in test
    years. The held-out group never contributes a single training row.
    """
    tm = temporal_split(df, cfg)
    for g in sorted(df[group_col].dropna().unique()):
        held = df[group_col] == g
        masks = {
            "train": tm["train"] & ~held,
            "calib": tm["calib"] & ~held,
            "test": tm["test"] & held,
        }
        assert_spatially_disjoint(df, masks)
        yield g, masks


def assert_spatially_disjoint(
    df: pd.DataFrame, masks: dict[str, pd.Series], unit: str = "geocode"
) -> None:
    """Raise if a test unit also appears in train or calibration rows."""
    test_units = set(df.loc[masks["test"], unit])
    for k in ("train", "calib"):
        if test_units & set(df.loc[masks[k], unit]):
            raise AssertionError(f"test units leak into {k}")
