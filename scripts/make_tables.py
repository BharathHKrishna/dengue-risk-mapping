"""Render result CSVs in ``results/`` as markdown tables (``results/tables.md``).

The README and report copy their numbers from this file, so every reported
number traces back to a CSV written by a pipeline stage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"


def ci(df: pd.DataFrame, m: str, d: int = 3) -> pd.Series:
    """Format ``point [lo, hi]``."""
    return df.apply(lambda r: f"{r[m]:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]", axis=1)


def md(df: pd.DataFrame) -> str:
    """DataFrame to a GitHub markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def temporal() -> str:
    t = pd.read_csv(RES / "temporal_metrics.csv")
    out = pd.DataFrame(
        {
            "model": t["model"],
            "MAE": ci(t, "mae"),
            "RMSE": ci(t, "rmse"),
            "R2": ci(t, "r2"),
            "skill vs climatology": ci(t, "skill_vs_clim"),
            "Spearman": t["spearman"].map("{:.3f}".format),
            "alert F1": t["alert_f1"].map("{:.3f}".format),
            "alert AUC-PR": t["alert_auc_pr"].map("{:.3f}".format),
            "n": t["n"],
        }
    )
    s = pd.read_csv(RES / "temporal_seed_metrics.csv")
    seeds = (
        s.groupby("model")
        .agg(
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            skill_mean=("skill_vs_clim", "mean"),
            skill_sd=("skill_vs_clim", "std"),
        )
        .reset_index()
        .round(4)
    )
    y = pd.read_csv(RES / "temporal_by_year.csv").round(3)
    iv = pd.read_csv(RES / "temporal_intervals.csv").groupby("model").mean(numeric_only=True)
    iv = iv.drop(columns=["seed"]).reset_index().round(3)
    return "\n\n".join(
        [
            "### Temporal hold-out (train 2015 to 2021, calibrate 2022, test 2023 to 2025)",
            "Metrics on log1p(monthly incidence per 100k). Brackets: 95% municipality cluster "
            "bootstrap CI (1,000 resamples). LightGBM rows use the mean of 3 seeds.",
            md(out),
            "Seed spread (single-seed models):",
            md(seeds),
            "By test year:",
            md(y),
            "90% interval coverage on the test years (mean over 3 seeds):",
            md(iv),
        ]
    )


def spatial() -> str:
    t = pd.read_csv(RES / "spatial_metrics.csv")
    out = pd.DataFrame(
        {
            "held-out region": t["region"],
            "model": t["model"],
            "MAE": ci(t, "mae"),
            "skill vs pooled climatology": ci(t, "skill_vs_clim"),
            "Spearman": t["spearman"].map("{:.3f}".format),
            "alert AUC-PR": t["alert_auc_pr"].map("{:.3f}".format),
        }
    )
    if "skill_vs_local_clim" in t:
        out["skill vs local climatology"] = t["skill_vs_local_clim"].map(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"
        )
    iv = pd.read_csv(RES / "spatial_intervals.csv").groupby("region").mean(numeric_only=True)
    iv = iv.drop(columns=["seed"]).reset_index().round(3)
    return "\n\n".join(
        [
            "### Spatial blocked CV (hold out one macro-region, train on the others; "
            "test years 2023 to 2025)",
            md(out),
            "90% interval coverage in the held-out region:",
            md(iv),
        ]
    )


def transfer() -> str:
    s = pd.read_csv(RES / "transfer_india_summary.csv").round(3)
    y = pd.read_csv(RES / "transfer_india_by_year.csv").round(3)
    parts = [
        "### Brazil to India transfer (state-year ranks vs reported cases)",
        md(s),
        "By year (Spearman with 95% state bootstrap CI):",
        md(y),
    ]
    f = RES / "transfer_india_seasonality.csv"
    if f.exists():
        parts += ["Seasonality check vs India national monthly cases:", md(pd.read_csv(f).round(3))]
    return "\n\n".join(parts)


def shap() -> str:
    parts = ["### SHAP"]
    for name in ("lgbm_climate", "lgbm_ar"):
        f = RES / f"shap_lag_importance_{name}.csv"
        if f.exists():
            parts += [f"Mean abs SHAP by climate lag, {name}:", md(pd.read_csv(f).round(4))]
        f = RES / f"shap_sanity_{name}.csv"
        if f.exists():
            parts += [f"Biological sanity checks, {name}:", md(pd.read_csv(f).round(3))]
    return "\n\n".join(parts)


def main() -> None:
    sections = []
    for fn in (temporal, spatial, shap, transfer):
        try:
            sections.append(fn())
        except FileNotFoundError as e:
            sections.append(f"({fn.__name__}: missing {Path(e.filename).name})")
    (RES / "tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print((RES / "tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
