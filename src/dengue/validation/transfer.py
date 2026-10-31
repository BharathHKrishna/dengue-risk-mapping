"""Brazil to India transfer test at state-year level.

The Brazil-trained climate-only model predicts monthly log incidence for each
Indian district. District predictions are converted to incidence, summed over
the year, and population-weighted to states. Predicted state ranks are then
compared with reported (OpenDengue/NCVBDC) incidence ranks for each year.

This is an out-of-distribution test: different climate regimes, different
surveillance systems and strong under-reporting in India. Only ranks are
compared, never absolute levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dengue.ingest.opendengue import MERGE_FOR_VALIDATION


def state_year_predictions(district_pred: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate district-month predictions to state-year predicted incidence per 100k.

    ``district_pred``: district_id, month, pred (log1p incidence per 100k per month).
    ``districts``: district_id, state_current, pop.
    """
    d = district_pred.merge(districts[["district_id", "state_current", "pop"]], on="district_id")
    d["state"] = d["state_current"].replace(MERGE_FOR_VALIDATION)
    d["year"] = d["month"].dt.year
    d["exp_cases"] = np.expm1(d["pred"]).clip(lower=0) * d["pop"] / 1e5
    annual = d.groupby(["state", "year"], as_index=False).agg(
        pred_cases=("exp_cases", "sum"), n_months=("month", "nunique")
    )
    pop = (
        districts.assign(state=districts["state_current"].replace(MERGE_FOR_VALIDATION))
        .groupby("state", as_index=False)["pop"]
        .sum()
    )
    annual = annual.merge(pop, on="state")
    annual["pred_incidence"] = annual["pred_cases"] / annual["pop"] * 1e5
    return annual


def rank_metrics(
    df: pd.DataFrame,
    score: str,
    truth: str = "obs_incidence",
    top_k: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Spearman rho (with state-bootstrap CI) and top-k overlap for one year."""
    d = df[[score, truth]].dropna()
    rho = float(spearmanr(d[score], d[truth]).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        if s[score].nunique() > 1 and s[truth].nunique() > 1:
            boots.append(spearmanr(s[score], s[truth]).statistic)
    top_pred = set(d[score].nlargest(top_k).index)
    top_obs = set(d[truth].nlargest(top_k).index)
    return {
        "n_states": len(d),
        "spearman": rho,
        "spearman_lo": float(np.nanquantile(boots, 0.025)),
        "spearman_hi": float(np.nanquantile(boots, 0.975)),
        f"top{top_k}_overlap": len(top_pred & top_obs),
    }


def transfer_table(
    annual_pred: pd.DataFrame,
    observed: pd.DataFrame,
    static_state: pd.DataFrame,
    years: list[int],
    min_cases: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-year rank comparison of the model and two simple reference rankings.

    References: (a) population density alone, (b) the previous reported year
    (persistence, which uses past case data the model never sees).
    Returns (metrics per year and method, merged state-year table).
    """
    obs = observed.assign(state=observed["state"].replace(MERGE_FOR_VALIDATION))
    obs = obs.groupby(["state", "year"], as_index=False)["cases"].sum()
    m = annual_pred.merge(obs, on=["state", "year"], how="inner")
    m["obs_incidence"] = m["cases"] / m["pop"] * 1e5
    m = m.merge(static_state, on="state", how="left")
    prev = m[["state", "year", "obs_incidence"]].assign(year=m["year"] + 1)
    m = m.merge(
        prev.rename(columns={"obs_incidence": "prev_obs_incidence"}),
        on=["state", "year"],
        how="left",
    )
    rows = []
    for yr in years:
        g = m[(m["year"] == yr) & (m["cases"] >= min_cases)].set_index("state")
        if len(g) < 5:
            continue
        for method, col in [
            ("model (Brazil-trained, climate only)", "pred_incidence"),
            ("population density only", "pop_density_w"),
            ("previous year reported (persistence)", "prev_obs_incidence"),
        ]:
            if g[col].notna().sum() >= 5:
                rows.append({"year": yr, "method": method, **rank_metrics(g, col)})
    return pd.DataFrame(rows), m
