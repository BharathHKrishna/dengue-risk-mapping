"""OpenDengue V1.3 (CC BY 4.0): India state-year dengue cases.

OpenDengue's India Admin1 records are compiled from NCVBDC annual reports.
State names vary across years (for example ORISSA and ODISHA), so they are
harmonized to the DataMeet ``States/Admin2`` names. Units that were merged or
split are handled as follows:

- Dadra and Nagar Haveli plus Daman and Diu are summed into the merged UT.
- Jammu and Kashmir is reported as one unit (Ladakh is not reported
  separately), so for validation Ladakh is merged into Jammu and Kashmir.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from dengue.ingest.india_boundaries import normalize_name
from dengue.utils.cache import download
from dengue.utils.config import path

URL = (
    "https://github.com/OpenDengue/master-repo/raw/main/data/releases/V1.3/Spatial_extract_V1_3.zip"
)

ALIASES = {
    "a and n island": "Andaman & Nicobar",
    "aandn island": "Andaman & Nicobar",
    "andaman and nicobar": "Andaman & Nicobar",
    "arunachal pradesh": "Arunachal Pradesh",
    "d and n haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "dandn haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "j and k": "Jammu & Kashmir",
    "jammu and kashmir": "Jammu & Kashmir",
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "uttrakhand": "Uttarakhand",
}

# State whose validation unit absorbs another (reporting does not split them).
MERGE_FOR_VALIDATION = {"Ladakh": "Jammu & Kashmir"}


def harmonize_state(name: str, valid: set[str]) -> str | None:
    """Map an OpenDengue adm_1_name to a DataMeet state name, or None if unknown."""
    n = normalize_name(name)
    if n in ALIASES:
        return ALIASES[n]
    lookup = {normalize_name(v): v for v in valid}
    return lookup.get(n)


def india_state_year(cfg: dict, valid_states: set[str]) -> pd.DataFrame:
    """Tidy India state-year cases: state, year, cases. Unmatched names raise."""
    out = path(cfg, "interim", "india_state_year_cases.parquet")
    if out.exists():
        return pd.read_parquet(out)
    raw = Path(cfg["paths"]["raw"])
    z = download(URL, raw / "opendengue" / "Spatial_extract_V1_3.zip", raw, note="OpenDengue V1.3")
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as f:
            df = pd.read_csv(f, low_memory=False)
    ind = df[(df["adm_0_name"] == "INDIA") & (df["S_res"] == "Admin1") & (df["T_res"] == "Year")]
    ind = ind.assign(year=pd.to_datetime(ind["calendar_start_date"]).dt.year)
    ind["state"] = ind["adm_1_name"].map(lambda s: harmonize_state(s, valid_states))
    unknown = ind.loc[ind["state"].isna(), "adm_1_name"].unique()
    if len(unknown):
        raise ValueError(f"unmatched OpenDengue India names: {unknown}")
    tidy = ind.groupby(["state", "year"], as_index=False)["dengue_total"].sum()
    tidy = tidy.rename(columns={"dengue_total": "cases"})
    tidy.to_parquet(out)
    return tidy


def india_national_monthly(cfg: dict) -> pd.DataFrame:
    """India national monthly cases (OpenDengue Admin0 Month records): month, cases."""
    raw = Path(cfg["paths"]["raw"])
    z = download(URL, raw / "opendengue" / "Spatial_extract_V1_3.zip", raw, note="OpenDengue V1.3")
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as f:
            df = pd.read_csv(f, low_memory=False)
    ind = df[(df["adm_0_name"] == "INDIA") & (df["S_res"] == "Admin0") & (df["T_res"] == "Month")]
    out = ind.assign(
        month=pd.to_datetime(ind["calendar_start_date"]).dt.to_period("M").dt.to_timestamp()
    )
    return (
        out.groupby("month", as_index=False)["dengue_total"]
        .sum()
        .rename(columns={"dengue_total": "cases"})
    )
