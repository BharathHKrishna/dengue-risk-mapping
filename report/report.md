---
title: "Does a satellite and climate based dengue model transfer to places it has never seen? Evidence from Brazil and India"
author: "Bharath Harikrishna"
date: "October 2026"
---

# Abstract

<!-- ABSTRACT -->

# 1. Introduction

Dengue is transmitted by *Aedes* mosquitoes, whose development, survival and
biting rates depend on temperature, rainfall, humidity and the availability of
standing water, with transmission typically lagging weather by one to three
months [Mordecai et al. 2017; Lowe et al. 2018]. Climate driven risk maps are
therefore attractive for health departments that need to plan vector control,
awareness campaigns and hospital capacity ahead of the season.

Most published dengue models are evaluated in the same places they were
trained on, often with random or temporal splits. A model deployed in a new
country meets different climates, cities and surveillance systems. This work
asks a narrower and more practical question: **how much of a climate and
environment model's skill survives when it is moved to regions it never saw?**

We train on Brazil, which publishes weekly notified dengue cases for all 5,570
municipalities through InfoDengue [Codeco et al. 2018], and test transfer in
two steps: (1) within Brazil, by holding out each of the five macro-regions,
and (2) to India, where only state-year counts are publicly available through
OpenDengue [Clarke et al. 2024]. We report skill against seasonal baselines,
calibrated uncertainty intervals, and SHAP explanations checked against known
dengue biology.

Contributions:

1. A reproducible public-data pipeline that aligns InfoDengue, OpenDengue and
   ten Google Earth Engine layers into municipality-month and district-month
   tables.
2. A validation protocol with temporal hold-out, spatial blocked
   cross-validation (blocked in both space and time), and a cross-country rank
   test, all with municipality cluster bootstrap confidence intervals.
3. A separation of what climate contributes from what past case counts and a
   place's own seasonal history contribute.
4. An open, monthly updated map that labels where predictions are and are not
   validated.

# 2. Data

**Labels.** InfoDengue weekly notified cases (`casos`) for 2014 to 2025 were
pulled for every municipality (627 epidemiological weeks each; no missing
weeks). Each week was assigned to the calendar month containing its Wednesday
and counts were scaled to a standard month of 4.345 weeks. Incidence uses the
InfoDengue (IBGE) population. India labels are OpenDengue V1.3 Admin1 annual
totals compiled from NCVBDC reports, harmonized to current state names
(Odisha and Orissa, the merged Dadra and Nagar Haveli and Daman and Diu, and
Ladakh merged into Jammu and Kashmir, which is reported as one unit).

**Predictors.** From Google Earth Engine: ERA5-Land monthly aggregates (mean,
minimum and maximum 2 m temperature; relative humidity from dewpoint by the
Magnus formula; total precipitation), CHIRPS daily rainfall summed per month,
MODIS MOD11A2 daytime land surface temperature, MODIS MOD13Q1 NDVI, JRC Global
Surface Water occurrence, GHSL 2020 population density and built surface,
VIIRS 2022 night lights, SRTM elevation, and ESA WorldCover class shares.
Monthly layers were reduced to administrative polygons at 5 km, static layers
at 1 km and land cover at 300 m; units smaller than a pixel were sampled at an
interior point.

**Boundaries.** IBGE 2022 municipalities for Brazil. For India, DataMeet state
and Census 2011 district layers, which follow the Survey of India official
external boundary; 2011 districts were assigned to current states by a
point-in-polygon join.

# 3. Methods

**Target and features.** The regression target is
$y = \log(1 + \text{incidence per } 10^5)$ per municipality-month. Features
are seven climate variables at lags of 1, 2 and 3 months, a cyclic month of
year, and 14 static covariates. A prediction for month $t$ uses only data
from months up to $t-1$; this is enforced by joining lags on explicit month
offsets and checked by unit tests that perturb all future data and assert that
earlier features are unchanged. An alert label marks months above a
municipality's own 75th percentile of $y$, with thresholds estimated on
training years only.

**Models.** Two baselines: local seasonal climatology (mean $y$ for that
municipality and month of year over training years) and the same month last
year. LightGBM gradient boosting (800 trees, learning rate 0.05, 63 leaves) in
two variants: `lgbm_climate` without any case history, and `lgbm_ar` that adds
$y$ at lags 1, 2, 3 and 12 months. Each learned model is fit with three seeds
and the point prediction is the seed mean.

**Uncertainty.** 5% and 95% LightGBM quantile models are conformalized
(CQR, Romano et al. 2019) on a held-out calibration year, giving nominal 90%
intervals. We report empirical coverage on test data.

**Validation.** (i) Temporal: train 2015 to 2021, calibrate 2022, test 2023 to
2025. (ii) Spatial blocked CV: for each macro-region (North, Northeast,
Southeast, South, Center-West), train on the other four regions in 2015 to
2021, calibrate on the other four in 2022, and test on the held-out region in
2023 to 2025. A held-out place therefore contributes no training row and no
recent history. (iii) Transfer: the deployment model (`lgbm_climate`, trained
2015 to 2024, calibrated 2025) predicts every Indian district-month; expected
cases are summed to state-years and compared by Spearman rank correlation and
top-5 overlap with reported incidence for 2015 to 2024. Two reference
rankings are reported: population density alone, and the previous reported
year (persistence).

**Metrics.** MAE and RMSE on $y$, $R^2$, MSE skill score against the
climatology baseline ($1 - \text{MSE}/\text{MSE}_\text{clim}$), Spearman
correlation, and for alerts precision, recall, F1 and AUC-PR. 95% confidence
intervals come from a cluster bootstrap that resamples whole municipalities
(1,000 resamples), because months within a municipality are correlated.

**Explainability.** SHAP values (TreeExplainer) on 20,000 test rows; binned
dependence curves are compared with the expected unimodal temperature
response (optimum near 26 to 29 deg C) and positive lagged rainfall effects.

# 4. Results

<!-- RESULTS -->

# 5. Discussion

<!-- DISCUSSION -->

# 6. Limitations

<!-- LIMITATIONS -->

# References

- Clarke J. et al. (2024). A global dataset of publicly available dengue case count data. *Scientific Data* 11, 296.
- Codeco C. et al. (2018). Infodengue: a nowcasting system for the surveillance of arboviruses in Brazil. *Rev. Epidemiol. Sante Publique* 66, S386.
- Funk C. et al. (2015). The climate hazards infrared precipitation with stations. *Scientific Data* 2, 150066.
- Lowe R. et al. (2018). Nonlinear and delayed impacts of climate on dengue risk in Barbados. *PLoS Medicine* 15, e1002613.
- Mordecai E. A. et al. (2017). Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models. *PLoS NTDs* 11, e0005568.
- Munoz Sabater J. (2019). ERA5-Land monthly averaged data from 1981 to present. Copernicus Climate Change Service.
- Pekel J.-F. et al. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature* 540, 418.
- Romano Y., Patterson E., Candes E. (2019). Conformalized quantile regression. *NeurIPS*.
- Schiavina M. et al. (2023). GHS-POP R2023A. European Commission, JRC.
- Lundberg S., Lee S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
