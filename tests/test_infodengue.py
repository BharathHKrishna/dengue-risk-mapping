"""Tests for the InfoDengue parser and weekly to monthly aggregation."""

import pandas as pd
import pytest

from dengue.ingest.infodengue import parse_alertcity, se_to_week_start, weekly_to_monthly

CSV = """data_iniSE,SE,casos_est,casos,p_inc100k,pop,nivel,tempmed
2023-12-24,202352,12.0,10,1.5,100000,2,27.1
2023-12-17,202351,8.0,8,1.2,100000,1,26.3
2023-12-17,202351,8.0,8,1.2,100000,1,26.3
"""


def test_parse_csv_maps_columns_and_dedups():
    df = parse_alertcity(CSV, 1234567)
    assert list(df["se"]) == [202351, 202352]
    assert df["cases"].tolist() == [8, 10]
    assert {"geocode", "week_start", "pop", "alert_level", "tempmed"} <= set(df.columns)
    assert (df["geocode"] == 1234567).all()


def test_parse_lowercase_json_style_columns():
    raw = pd.DataFrame({"se": [202301], "casos": [3], "pop": [5000], "data_iniSE": ["2023-01-01"]})
    df = parse_alertcity(raw, 1)
    assert df.loc[0, "cases"] == 3
    assert df.loc[0, "week_start"] == pd.Timestamp("2023-01-01")


def test_parse_missing_required_raises():
    with pytest.raises(ValueError):
        parse_alertcity("SE,casos\n202301,1\n", 1)


def test_parse_empty_response():
    assert parse_alertcity("", 1).empty


def test_se_to_week_start_matches_known_weeks():
    # Known Brazilian epi week starts (from InfoDengue data_iniSE).
    se = pd.Series([202301, 202352, 201401])
    got = se_to_week_start(se).tolist()
    expected = ["2023-01-01", "2023-12-24", "2013-12-29"]
    assert got == [pd.Timestamp(d) for d in expected]


def test_weekly_to_monthly_assigns_each_week_once():
    weeks = pd.date_range("2023-01-01", periods=9, freq="7D")
    w = pd.DataFrame({"geocode": 1, "week_start": weeks, "cases": 1, "pop": 1000})
    m = weekly_to_monthly(w)
    assert m["cases"].sum() == 9
    # 2023-01-29 has its Wednesday on Feb 1, so it goes to February.
    assert m.set_index("month").loc[pd.Timestamp("2023-01-01"), "n_weeks"] == 4
