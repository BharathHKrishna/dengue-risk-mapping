"""Extra analyses computed from saved pipeline outputs (no retraining).

Writes:
- results/temporal_level_by_year.csv: mean observed vs predicted log incidence
  and 90% interval coverage per test year (shows the epidemic level shift).
- results/train_level_by_year.csv: mean observed log incidence per year.
- results/india_out_of_range.csv: share of India district-months with inputs
  outside the Brazil training range, and the most frequent such inputs.
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
    import json
    from collections import Counter

    dp = pd.read_parquet(ROOT / "app" / "data" / "india_district_predictions.parquet")
    flags = dp["ood"].map(json.loads)
    any_ood = flags.map(len) > 0
    counts = Counter(f for fl in flags for f in fl)
    rows = [
        {"item": "district-months", "value": len(dp)},
        {
            "item": "share with any input outside Brazil training range",
            "value": round(any_ood.mean(), 4),
        },
        {
            "item": "districts affected in at least one month",
            "value": dp.loc[any_ood, "region_id"].nunique(),
        },
        {"item": "districts total", "value": dp["region_id"].nunique()},
    ] + [{"item": f"rows with {k} out of range", "value": v} for k, v in counts.most_common(8)]
    pd.DataFrame(rows).to_csv(RES / "india_out_of_range.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print(agg.round(3).to_string())
    print(lv.round(3).to_string())


if __name__ == "__main__":
    main()
