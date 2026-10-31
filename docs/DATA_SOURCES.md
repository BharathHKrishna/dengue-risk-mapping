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

| DataMeet `States/Admin2` (github.com/datameet/maps) | India states and UTs (36), drawn to the Survey of India official external boundary | CC BY 4.0 (repository default) | verified 2026-10-31 |
| DataMeet `Districts/Census_2011/2011_Dist` | India districts (641, Census 2011) | CC BY 2.5 India | verified 2026-10-31 |

**India boundary note (brief rule 8):** GAUL and geoBoundaries draw de facto
lines in Jammu and Kashmir and Arunachal Pradesh that do not match India's
official map, so they are not used. The DataMeet layers follow the Survey of
India external boundary. Limitations: districts are the 2011 Census set (later
splits are not shown), and 2011 districts are assigned to current states
(Telangana, Ladakh) by a point-in-polygon join. Attribution: "India boundaries
by DataMeet India community (CC BY 4.0 / CC BY 2.5 IN)".

## Predictors (Google Earth Engine)

All asset IDs were verified with `ee.data.getAsset` on 2026-10-31
(`dengue.ingest.gee_extract.verify_assets`). Monthly climate is reduced at
5 km, static layers at 1 km, land cover at 300 m. Units smaller than one pixel
are sampled at an interior point.

| Variable(s) | Asset ID | Processing | License / citation |
|---|---|---|---|
| t2m_mean, t2m_min, t2m_max, rh_mean, precip_era5 | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | monthly mean, min, max 2 m temperature (deg C); RH from 2 m dewpoint by the Magnus formula; total precipitation (mm) | Copernicus licence; Munoz Sabater J. (2019) ERA5-Land monthly data, C3S CDS. Contains modified Copernicus Climate Change Service information. |
| precip_chirps | `UCSB-CHG/CHIRPS/DAILY` | monthly sum of daily rainfall (mm) | Public domain (CC0 waiver); Funk C. et al. (2015), Scientific Data 2, 150066 |
| lst_day | `MODIS/061/MOD11A2` | monthly mean of 8-day daytime LST (deg C) | NASA LP DAAC, no restrictions; Wan Z., Hook S., Hulley G. (2021) MOD11A2 v061 |
| ndvi_mean | `MODIS/061/MOD13Q1` | mean NDVI 2019 to 2021 | NASA LP DAAC, no restrictions; Didan K. (2021) MOD13Q1 v061 |
| water_occurrence | `JRC/GSW1_4/GlobalSurfaceWater` | mean occurrence 1984 to 2021 (0 to 1) | Copernicus / JRC, free use with attribution; Pekel J.-F. et al. (2016), Nature 540, 418 |
| pop_density_log | `JRC/GHSL/P2023A/GHS_POP` (2020 epoch) | persons per km2 (count x 100 on the 100 m equal-area grid), nodata masked, log1p | CC BY 4.0; Schiavina M. et al. (2023) GHS-POP R2023A, JRC |
| built_frac | `JRC/GHSL/P2023A/GHS_BUILT_S` (2020 epoch) | built-up surface share (m2 / 10,000), nodata masked | CC BY 4.0; Pesaresi M., Politis P. (2023) GHS-BUILT-S R2023A, JRC |
| lc_* (8 classes) | `ESA/WorldCover/v200` | class share per unit at 300 m | CC BY 4.0; Zanaga D. et al. (2022) ESA WorldCover 10 m 2021 v200 |
| ntl_mean | `NOAA/VIIRS/DNB/ANNUAL_V22` (2022) | annual average radiance | CC BY 4.0; Elvidge C.D. et al. (2021), Remote Sensing 13, 922. The collection starts in 2022, so 2022 is used, not 2020. |
| elevation | `USGS/SRTMGL1_003` | mean elevation (m) | Public domain; Farr T.G. et al. (2007), Rev. Geophys. 45 |

Population for incidence: Brazil uses the IBGE estimate shipped by InfoDengue
(`pop`). India state and district population is the GHSL 2020 density times
unit area (one epoch for all years, which is a limitation for 2015 to 2024).
