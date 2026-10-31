"""Extra analyses computed from saved pipeline outputs (no retraining).

Writes:
- results/temporal_level_by_year.csv: mean observed vs predicted log incidence
  and 90% interval coverage per test year (shows the epidemic level shift).
- results/train_level_by_year.csv: mean observed log incidence per year.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> None:
    p = pd.read_parquet(RES / "temporal_test_predictions.parquet")
    p["year"] = p["month"].dt.year
    out = {}
    for m in ("lgbm_climate", "lgbm_ar"):
        p[f"inside_{m}"] = (p["y"] >= p[f"lo_{m}"]) & (p["y"] <= p[f"hi_{m}"])
    agg = p.groupby("year").agg(
        observed_mean_y=("y", "mean"),
        climatology_mean=("clim", "mean"),
        lgbm_climate_mean=("pred_lgbm_climate", "mean"),
        lgbm_ar_mean=("pred_lgbm_ar", "mean"),
        coverage_lgbm_climate=("inside_lgbm_climate", "mean"),
        coverage_lgbm_ar=("inside_lgbm_ar", "mean"),
    )
    agg.round(4).to_csv(RES / "temporal_level_by_year.csv")
    out["level"] = agg
    d = pd.read_parquet(ROOT / "data" / "processed" / "brazil_monthly.parquet")
    lv = d.groupby("year").agg(mean_y=("y", "mean"), notified_cases=("cases", "sum"))
    lv.round(4).to_csv(RES / "train_level_by_year.csv")
    print(agg.round(3).to_string())
    print(lv.round(3).to_string())


if __name__ == "__main__":
    main()
