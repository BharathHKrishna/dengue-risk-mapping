"""Leakage tests: lag features, split logic, label thresholds."""

import numpy as np
import pandas as pd
import pytest

from dengue.features.build import add_high_label, add_lags
from dengue.validation.splits import (
    assert_temporal_order,
    spatial_folds,
    temporal_split,
)

CFG = {
    "split": {"train_years": [2015, 2016], "calib_years": [2017, 2017], "test_years": [2018, 2019]}
}


def panel(n_units: int = 6, start: str = "2015-01", months: int = 60) -> pd.DataFrame:
    t = pd.date_range(start, periods=months, freq="MS")
    df = pd.DataFrame([(g, m) for g in range(n_units) for m in t], columns=["geocode", "month"])
    df["year"] = df["month"].dt.year
    # Value encodes the month index so lags can be checked exactly.
    df["v"] = (df["month"].dt.year - 2015) * 12 + df["month"].dt.month
    df["region"] = np.where(df["geocode"] < 3, "A", "B")
    return df


def test_lag_features_only_use_past_months():
    df = add_lags(panel(), ["v"], [1, 2, 3])
    for k in (1, 2, 3):
        ok = df[f"v_lag{k}"].dropna()
        assert (ok == df.loc[ok.index, "v"] - k).all()
        # Each lag value is strictly earlier than the row's own month.
        assert (df.loc[ok.index, f"v_lag{k}"] < df.loc[ok.index, "v"]).all()


def test_lags_respect_gaps_and_units():
    df = panel(n_units=2)
    df = df[~((df["geocode"] == 0) & (df["month"] == "2016-03-01"))]
    out = add_lags(df, ["v"], [1])
    row = out[(out["geocode"] == 0) & (out["month"] == "2016-04-01")]
    assert row["v_lag1"].isna().all(), "a missing month must not be filled by an older one"
    first = out[out["month"] == out["month"].min()]
    assert first["v_lag1"].isna().all(), "first month has no past, lag must be NaN"


def test_lag_zero_is_rejected():
    with pytest.raises(ValueError):
        add_lags(panel(), ["v"], [0])


def test_future_values_do_not_change_past_features():
    """Changing all future data must leave features at earlier months unchanged."""
    df = panel()
    cut = pd.Timestamp("2018-01-01")
    a = add_lags(df, ["v"], [1, 2, 3])
    df2 = df.copy()
    df2.loc[df2["month"] >= cut, "v"] = -999
    b = add_lags(df2, ["v"], [1, 2, 3])
    cols = ["v_lag1", "v_lag2", "v_lag3"]
    mask = a["month"] <= cut  # month == cut uses only months < cut
    pd.testing.assert_frame_equal(a.loc[mask, cols], b.loc[mask, cols])


def test_temporal_split_is_ordered_and_disjoint():
    df = panel()
    m = temporal_split(df, CFG)
    assert df.loc[m["train"], "month"].max() < df.loc[m["calib"], "month"].min()
    assert df.loc[m["calib"], "month"].max() < df.loc[m["test"], "month"].min()
    assert not (m["train"] & m["test"]).any()


def test_temporal_order_check_catches_overlap():
    df = panel()
    bad = {"train": df["year"] <= 2018, "calib": df["year"] == 2017, "test": df["year"] >= 2018}
    with pytest.raises(AssertionError):
        assert_temporal_order(df, bad)


def test_spatial_folds_hold_out_whole_regions_in_space_and_time():
    df = panel()
    folds = list(spatial_folds(df, CFG))
    assert [g for g, _ in folds] == ["A", "B"]
    for g, m in folds:
        assert (df.loc[m["test"], "region"] == g).all()
        assert not (df.loc[m["train"], "region"] == g).any()
        assert not (df.loc[m["calib"], "region"] == g).any()
        assert df.loc[m["train"], "month"].max() < df.loc[m["test"], "month"].min()


def test_high_threshold_uses_training_rows_only():
    df = panel(n_units=1)
    df["y"] = np.where(df["year"] <= 2016, 1.0, 100.0)
    df["cases"] = 5
    train = df["year"] <= 2016
    out = add_high_label(df, train, q=0.75)
    assert out["high_thr"].iloc[0] == 1.0
    # Changing test-period labels must not move the threshold.
    df.loc[~train, "y"] = 0.0
    assert add_high_label(df, train, q=0.75)["high_thr"].iloc[0] == 1.0
