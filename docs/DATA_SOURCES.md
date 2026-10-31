# Data sources

Every source used in this repo, with license, citation and how it is accessed.
Raw pulls are cached under `data/raw/` and logged with UTC pull dates in
`data/raw/manifest.jsonl`. Earth Engine extractions record their run date in
the output Parquet metadata.

Status column: "verified" means the endpoint or asset was reached from this
repo's code on the date given.

## Case labels

| Source | Use | Access | License | Status |
|---|---|---|---|---|
| InfoDengue (Fiocruz / FGV) | Brazil weekly municipal dengue cases, population | `https://info.dengue.mat.br/api/alertcity` | Open data, attribution requested; see terms at https://info.dengue.mat.br | verified 2026-10-30 |
| OpenDengue V1.3 | India state-year dengue cases (validation labels) | GitHub release zips (National, Spatial, Temporal extracts) | CC BY 4.0 | verified 2026-10-30 |

**InfoDengue citation:** Codeco C. et al. "Infodengue: A nowcasting system for the surveillance of arboviruses in Brazil." Revue d'Epidemiologie et de Sante Publique 66 (2018): S386. Data from the Brazilian notifiable diseases system (SINAN).

**OpenDengue citation:** Clarke J. et al. "A global dataset of publicly available dengue case count data." Scientific Data 11, 296 (2024). Version 1.3.

**OpenDengue India coverage (checked 2026-10-30, Spatial extract V1.3):**

| Level | Temporal resolution | Years | Units |
|---|---|---|---|
| Admin0 (national) | Year | 1991 to 2014 | 1 |
| Admin0 (national) | Month | 2024 to 2025 | 1 |
| Admin1 (state/UT) | Year | 1997 to 2012, 2015 to 2024 | 14 to 15 units (1997 to 2005), 17 (2006 to 2008), 33 (2009 to 2012), 35 to 36 (2015 to 2024) |
| Admin2 (district) | none | none | none |

There are no district or monthly state labels for India, so district maps are
unvalidated relative risk only. India state-year numbers in OpenDengue come from
NCVBDC reports. The dataful.in table was not used, so its terms do not apply.

## Boundaries

| Source | Use | License | Status |
|---|---|---|---|
| IBGE Malha Municipal 2022 (`BR_Municipios_2022.zip`, geoftp.ibge.gov.br) | Brazil municipalities (5,570, after dropping two lagoon pseudo-units) | Brazilian public government data, free reuse with attribution (IBGE) | verified 2026-10-30 |

The IBGE REST API (`servicodados.ibge.gov.br`) could not be reached from this
machine (connection timeout), so the official geoftp mesh is used instead.

India boundaries: added in Week 7 (see below).

## Predictors (Google Earth Engine)

Filled in Week 3, once each asset ID is verified in the catalog.
