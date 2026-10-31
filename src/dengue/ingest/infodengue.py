"""InfoDengue weekly municipal dengue data (Brazil).

API: https://info.dengue.mat.br/api/alertcity . One call per municipality,
cached as raw CSV under ``data/raw/infodengue/<geocode>.csv``.

Columns used (meaning per the InfoDengue API documentation):

- ``data_iniSE``: start date (Sunday) of the epidemiological week.
- ``SE``: epidemiological week as YYYYWW.
- ``casos``: notified cases in that week (as of the pull date). This is the label.
- ``casos_est``: nowcast-corrected estimate for recent weeks. Not used as label,
  because it is model output, not an observation.
- ``p_inc100k``: estimated incidence per 100k (based on ``casos_est``).
- ``pop``: municipal population estimate used by InfoDengue (IBGE).
- ``nivel``: InfoDengue alert level 1 to 4.
- ``tempmin``, ``tempmed``, ``tempmax``, ``umidmin``, ``umidmed``, ``umidmax``:
  weekly climate summaries computed by InfoDengue (kept only as a cross-check).
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from dengue.utils.cache import record_pull
from dengue.utils.config import path
from dengue.utils.log import get_logger

log = get_logger(__name__)

API = "https://info.dengue.mat.br/api/alertcity"

# Canonical name -> accepted aliases (lower-case) seen across CSV and JSON outputs.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "week_start": ("data_inise", "data_ini_se", "data_inicio_se"),
    "se": ("se", "semana_epi", "ew"),
    "cases": ("casos",),
    "cases_est": ("casos_est",),
    "inc100k_est": ("p_inc100k",),
    "pop": ("pop", "populacao"),
    "alert_level": ("nivel",),
    "tempmin": ("tempmin",),
    "tempmed": ("tempmed",),
    "tempmax": ("tempmax",),
    "umidmin": ("umidmin",),
    "umidmed": ("umidmed",),
    "umidmax": ("umidmax",),
}
REQUIRED = ("se", "cases", "pop")


def build_url(geocode: int, disease: str, ey_start: int, ey_end: int) -> str:
    """Build the alertcity URL. All parameters are required by the API."""
    return (
        f"{API}?geocode={geocode}&disease={disease}&format=csv"
        f"&ew_start=1&ew_end=53&ey_start={ey_start}&ey_end={ey_end}"
    )


def parse_alertcity(text_or_df: str | pd.DataFrame, geocode: int) -> pd.DataFrame:
    """Parse an alertcity response (CSV text or a DataFrame from JSON) into a tidy table.

    Robust to column name case and the known aliases. Returns one row per
    epidemiological week with columns: geocode, se, week_start, cases, pop and
    any available optional columns. Raises ``ValueError`` if a required column
    is missing.
    """
    if isinstance(text_or_df, str):
        if not text_or_df.strip():
            return pd.DataFrame(columns=["geocode", *REQUIRED, "week_start"])
        df = pd.read_csv(io.StringIO(text_or_df))
    else:
        df = text_or_df.copy()
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = canon
                break
    missing = [c for c in REQUIRED if c not in rename.values()]
    if missing:
        raise ValueError(f"alertcity response for {geocode} lacks columns {missing}")
    df = df[list(rename)].rename(columns=rename)
    df["se"] = df["se"].astype(int)
    if "week_start" in df:
        df["week_start"] = pd.to_datetime(df["week_start"])
    else:
        df["week_start"] = se_to_week_start(df["se"])
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce")
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
    df.insert(0, "geocode", int(geocode))
    df = df.drop_duplicates("se").sort_values("se").reset_index(drop=True)
    return df


def se_to_week_start(se: pd.Series) -> pd.Series:
    """Approximate the Sunday start date of epi week YYYYWW (fallback only).

    Epi week 1 is the Sunday-to-Saturday week that contains January 4.
    """
    year = se // 100
    week = se % 100

    def start(y: int) -> pd.Timestamp:
        jan4 = pd.Timestamp(year=y, month=1, day=4)
        # weekday(): Monday=0 ... Sunday=6. Step back to the Sunday on or before Jan 4.
        return jan4 - pd.Timedelta(days=(jan4.weekday() + 1) % 7)

    starts = year.map(lambda y: start(int(y)))
    return starts + pd.to_timedelta((week - 1) * 7, unit="D")


def fetch_municipality(cfg: dict, geocode: int, session: requests.Session) -> Path | None:
    """Fetch and cache one municipality. Returns the cache path, or None on failure."""
    icfg = cfg["brazil"]["infodengue"]
    raw = Path(cfg["paths"]["raw"])
    dest = raw / "infodengue" / f"{geocode}.csv"
    if dest.exists():
        return dest
    url = build_url(geocode, icfg["disease"], icfg["ey_start"], icfg["ey_end"])
    for attempt in range(4):
        try:
            r = session.get(url, timeout=icfg["timeout_s"])
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so an interrupted run never leaves a truncated cache file.
            tmp = dest.with_suffix(".csv.part")
            tmp.write_text(r.text, encoding="utf-8")
            tmp.replace(dest)
            record_pull(raw, url, dest, note="InfoDengue alertcity")
            time.sleep(icfg["delay_s"])
            return dest
        except requests.RequestException as e:
            wait = 5 * 2**attempt
            log.warning("geocode %d failed (%s), retry in %ds", geocode, e, wait)
            time.sleep(wait)
    return None


def fetch_all(cfg: dict, geocodes: list[int]) -> list[int]:
    """Fetch all municipalities politely (cached). Returns the geocodes that failed."""
    failed = []
    with requests.Session() as s:
        s.headers["User-Agent"] = "dengue-risk-mapping research project"
        for i, g in enumerate(geocodes):
            if fetch_municipality(cfg, g, s) is None:
                failed.append(g)
            if i % 250 == 0:
                log.info("InfoDengue progress %d/%d", i, len(geocodes))
    return failed


def weekly_to_monthly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate epi weeks to calendar months.

    Each week is assigned to the month containing its Wednesday (start + 3 days),
    so every week belongs to exactly one month. Returns geocode, month
    (first day of month, Timestamp), cases (sum), pop (mean), n_weeks.
    """
    w = weekly.copy()
    w["month"] = (w["week_start"] + pd.Timedelta(days=3)).dt.to_period("M").dt.to_timestamp()
    m = w.groupby(["geocode", "month"], as_index=False).agg(
        cases=("cases", "sum"), pop=("pop", "mean"), n_weeks=("cases", "size")
    )
    return m


def build_weekly_table(cfg: dict, geocodes: list[int]) -> pd.DataFrame:
    """Parse all cached CSVs into one weekly Parquet table."""
    raw = Path(cfg["paths"]["raw"]) / "infodengue"
    parts = []
    for g in geocodes:
        f = raw / f"{g}.csv"
        if not f.exists():
            continue
        df = parse_alertcity(f.read_text(encoding="utf-8"), g)
        if len(df):
            parts.append(df)
    weekly = pd.concat(parts, ignore_index=True)
    weekly.to_parquet(path(cfg, "interim", "infodengue_weekly.parquet"))
    return weekly
