"""SHAP explanations and biological sanity checks for the climate-only model.

Checks against known Aedes and dengue biology:

- Temperature: transmission rises with temperature to an optimum (about
  26 to 29 deg C in lab and field studies, e.g. Mordecai et al. 2017) and
  falls above it. We look for a rising then flattening or falling SHAP curve.
- Rainfall: positive effect at 1 to 3 month lags (breeding sites).
- Elevation: negative effect (cooler at altitude).
"""

from __future__ import annotations

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from dengue.utils.config import path  # noqa: E402


def compute_shap(
    model_file: str, X: pd.DataFrame, n: int = 20000, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values for a random sample of rows. Returns (X_sample, shap_frame)."""
    booster = lgb.Booster(model_file=model_file)
    Xs = X.sample(min(n, len(X)), random_state=seed)
    sv = shap.TreeExplainer(booster).shap_values(Xs)
    return Xs, pd.DataFrame(sv, columns=Xs.columns, index=Xs.index)


def binned_effect(x: pd.Series, s: pd.Series, bins: int = 20) -> pd.DataFrame:
    """Mean SHAP value per quantile bin of the feature (a SHAP dependence curve)."""
    q = pd.qcut(x, bins, duplicates="drop")
    g = pd.DataFrame({"x": x, "s": s, "bin": q}).groupby("bin", observed=True)
    return g.agg(x_mid=("x", "median"), shap_mean=("s", "mean"), n=("s", "size")).reset_index(
        drop=True
    )


def sanity_checks(Xs: pd.DataFrame, sv: pd.DataFrame) -> pd.DataFrame:
    """Summarise direction and optimum of key climate effects from SHAP curves."""
    rows = []
    for f in [c for c in Xs.columns if c.startswith(("t2m_mean_lag", "lst_day_lag"))]:
        b = binned_effect(Xs[f], sv[f])
        peak = b.loc[b["shap_mean"].idxmax()]
        rows.append(
            {
                "feature": f,
                "check": "temperature optimum",
                "peak_x": float(peak["x_mid"]),
                "shap_at_min_x": float(b["shap_mean"].iloc[0]),
                "shap_at_peak": float(peak["shap_mean"]),
                "shap_at_max_x": float(b["shap_mean"].iloc[-1]),
                "spearman_x_shap": float(Xs[f].corr(sv[f], method="spearman")),
            }
        )
    for f in [c for c in Xs.columns if c.startswith(("precip_chirps_lag", "rh_mean_lag"))] + [
        "elevation"
    ]:
        if f in Xs:
            rows.append(
                {
                    "feature": f,
                    "check": "direction",
                    "spearman_x_shap": float(Xs[f].corr(sv[f], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def plots(cfg: dict, Xs: pd.DataFrame, sv: pd.DataFrame, tag: str) -> None:
    """Global importance bar chart and dependence curves for climate features."""
    imp = sv.abs().mean().sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP| (log1p incidence)")
    ax.set_title(f"Top 20 features by mean |SHAP|, {tag}")
    fig.tight_layout()
    fig.savefig(path(cfg, "results", "figures", f"shap_importance_{tag}.png"), dpi=150)
    plt.close(fig)

    feats = [
        "t2m_mean_lag1",
        "t2m_mean_lag2",
        "lst_day_lag1",
        "precip_chirps_lag1",
        "precip_chirps_lag2",
        "rh_mean_lag1",
        "elevation",
        "pop_density_log",
    ]
    feats = [f for f in feats if f in Xs]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, f in zip(axes.flat, feats, strict=False):
        ax.scatter(Xs[f], sv[f], s=2, alpha=0.15, color="#999999")
        b = binned_effect(Xs[f], sv[f])
        ax.plot(b["x_mid"], b["shap_mean"], color="#c0392b", lw=2)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f, fontsize=10)
        ax.set_ylabel("SHAP")
    fig.suptitle(f"SHAP dependence (red: binned mean), {tag}")
    fig.tight_layout()
    fig.savefig(path(cfg, "results", "figures", f"shap_dependence_{tag}.png"), dpi=150)
    plt.close(fig)


def lag_importance(sv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mean |SHAP| for each climate variable at each lag."""
    rows = []
    for v in cfg["features"]["climate_vars"]:
        for k in cfg["features"]["lags"]:
            c = f"{v}_lag{k}"
            if c in sv:
                rows.append({"variable": v, "lag": k, "mean_abs_shap": float(sv[c].abs().mean())})
    return pd.DataFrame(rows).pivot(index="variable", columns="lag", values="mean_abs_shap")


def top_drivers(sv_row: np.ndarray, cols: list[str], k: int = 3) -> list[dict]:
    """Top ``k`` features by |SHAP| for one prediction, signed."""
    idx = np.argsort(-np.abs(sv_row))[:k]
    return [{"feature": cols[i], "shap": round(float(sv_row[i]), 4)} for i in idx]
