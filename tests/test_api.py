"""API tests against a tiny synthetic data directory."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.api.main as m

    months = pd.to_datetime(["2025-11-01", "2025-12-01"])
    drivers = json.dumps(
        [["t2m_mean_lag1", 0.4], ["precip_chirps_lag2", -0.1], ["elevation", 0.05]]
    )
    pd.DataFrame(
        {
            "region_id": "BR-1",
            "month": months,
            "pred": [1.0, 2.0],
            "lo": [0.5, 1.0],
            "hi": [2.0, 3.0],
            "drivers": drivers,
            "source": "final model",
            "obs": [1.1, None],
            "cases": [3, None],
            "pop": 1000,
        }
    ).to_parquet(tmp_path / "brazil_predictions.parquet")
    pd.DataFrame(
        {
            "region_id": "IN-D001",
            "month": months,
            "pred": [0.2, 0.3],
            "lo": [0.1, 0.1],
            "hi": [0.5, 0.6],
            "drivers": drivers,
            "rel_risk_pct": [40.0, 60.0],
            "ood": ["[]", '["elevation"]'],
        }
    ).to_parquet(tmp_path / "india_district_predictions.parquet")
    pd.DataFrame(
        {
            "state": ["Kerala"],
            "year": [2024],
            "pred_incidence": [12.0],
            "obs_incidence": [50.0],
            "cases": [20000],
            "pop": [3.5e7],
        }
    ).to_parquet(tmp_path / "india_state_year.parquet")
    for f, pid, name in [
        ("brazil_municipalities.geojson", "BR-1", "A"),
        ("india_districts.geojson", "IN-D001", "B"),
        ("india_states.geojson", "IN-STATE-Kerala", None),
    ]:
        props = {"region_id": pid, "name": name, "state": "Kerala"}
        (tmp_path / f).write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "properties": props, "geometry": None}],
                }
            )
        )
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "climate_data_through": "2025-11",
                "latest_prediction_month": "2025-12",
                "brazil_cases_through": "2025-11",
                "india_cases_through": 2024,
                "generated_utc": "x",
                "interval": "90% conformal interval",
            }
        )
    )
    monkeypatch.setattr(m, "DATA", tmp_path)
    m.tables.cache_clear()
    m.meta.cache_clear()
    return TestClient(m.app)


def test_risk_brazil_returns_interval_drivers_and_dates(client):
    r = client.get("/risk", params={"region_id": "BR-1", "month": "2025-11"}).json()
    assert r["predicted_incidence_per_100k"] == pytest.approx(1.72, abs=0.01)
    assert r["interval_per_100k"][0] < r["predicted_incidence_per_100k"] < r["interval_per_100k"][1]
    assert r["top_drivers"][0]["label"] == "mean air temperature, 1 month earlier"
    assert r["data_dates"]["climate_data_through"] == "2025-11"
    assert "prototype" in r["disclaimer"]


def test_risk_brazil_month_without_observation(client):
    r = client.get("/risk", params={"region_id": "BR-1", "month": "2025-12"}).json()
    assert r["observed_incidence_per_100k"] is None


def test_risk_india_district_is_labeled_unvalidated(client):
    r = client.get("/risk", params={"region_id": "IN-D001", "month": "2025-12"}).json()
    assert r["relative_risk_percentile"] == 60.0
    assert r["inputs_outside_training_range"] == ["elevation"]
    assert "extrapolat" in r["extrapolation_warning"]
    assert "NOT validated" in r["disclaimer"]


def test_risk_india_state(client):
    r = client.get("/risk", params={"region_id": "IN-STATE-Kerala", "month": "2024-01"}).json()
    assert r["reported_cases"] == 20000


def test_map_and_months(client):
    assert client.get("/months", params={"view": "brazil"}).json()["months"] == [
        "2025-11",
        "2025-12",
    ]
    v = client.get("/map", params={"view": "brazil", "month": "2025-12"}).json()["values"]
    assert v["BR-1"][1] is None
    assert client.get("/map", params={"view": "india_states", "month": "2024"}).json()["values"]


def test_bad_requests(client):
    assert client.get("/risk", params={"region_id": "XX-1", "month": "2025-11"}).status_code == 400
    assert client.get("/risk", params={"region_id": "BR-1", "month": "2030-01"}).status_code == 404
    assert client.get("/risk", params={"region_id": "BR-1", "month": "bad"}).status_code == 422
