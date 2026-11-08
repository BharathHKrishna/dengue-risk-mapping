---
title: "Does a satellite and climate based dengue model transfer to places it has never seen? Evidence from Brazil and India"
author: "Bharath Harikrishna"
date: "October 2026"
---

# Abstract

Climate-driven dengue risk maps are often validated where they were
trained. We build a monthly model from ten public satellite and reanalysis
layers for all 5,570 Brazilian municipalities (2015 to 2025, InfoDengue
notified cases) and test it on time periods, regions and a country it never
saw. On the 2023 to 2025 hold-out, a climate-only LightGBM model barely beats
each municipality's own seasonal climatology (MSE skill 0.012, 95% CI 0.006
to 0.017), and given that climatology as an input, climate adds no skill
(-0.008). Past incidence lifts skill to 0.660, mostly through persistence.
When whole macro-regions are held out, the climate model beats the
history-free pooled baseline in four of five regions (skill up to 0.357) but
not local history. Transferred to India, it reproduces the national seasonal
cycle (Spearman 0.855 with 2024 monthly cases) yet ranks states worse than
population density alone (mean Spearman 0.146 vs 0.258). SHAP recovers the
expected unimodal temperature response (peak 25.8 to 27.5 deg C) and exposes
elevation as a Brazilian geography proxy that drove the largest India errors;
24.7% of Indian district-months have inputs outside the training range.
Conformal 90% intervals covered 0.74 to 0.98 across held-out settings. We
release the pipeline and a monthly updated map that labels where predictions
are and are not validated.

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

**Temporal hold-out** (Table 1; 200,520 test municipality-months). The
climate-only model reached MAE 1.821 (95% CI 1.799 to 1.843) and skill 0.012
(0.006 to 0.017) against local climatology (MAE 1.758). Last-year-same-month
had lower MAE (1.637) but negative skill (-0.040), and the autoregressive
model reached MAE 0.984 and skill 0.660 (0.654 to 0.667), Spearman 0.821 and
alert AUC-PR 0.806 (prevalence 0.464). Across the three seeds the standard
deviation of MAE was at most 0.001. By year, climate-only skill was -0.080,
0.086 and -0.037 for 2023, 2024 and 2025. Mean observed $y$ in those years was
2.23, 3.27 and 2.60, while the climate model predicted means of 1.40, 1.61
and 1.42.

| Table 1 | MAE | Skill vs climatology | Spearman | Alert AUC-PR |
|---|---|---|---|---|
| Local climatology | 1.758 [1.736, 1.781] | 0 | 0.576 | 0.515 |
| Last year, same month | 1.637 [1.621, 1.656] | -0.040 [-0.060, -0.021] | 0.540 | 0.627 |
| LightGBM, climate only | 1.821 [1.799, 1.843] | 0.012 [0.006, 0.017] | 0.602 | 0.520 |
| LightGBM + past cases | 0.984 [0.974, 0.995] | 0.660 [0.654, 0.667] | 0.821 | 0.806 |

**Value of climate beyond local history.** With the local climatology value
as an input, adding static covariates gave skill 0.000 and adding lagged
climate gave -0.008 (-0.010 to -0.006).

**Spatial blocked CV** (Table 2). Against the pooled climatology that is the
only option without local data, the climate model's skill was 0.357
(Center-West), -0.046 (North), 0.041 (Northeast), 0.085 (South) and 0.287
(Southeast). Against the held-out places' own history it was positive only in
the South (0.101).

| Table 2: held-out region | Skill vs pooled climatology | Skill vs own history | 90% coverage |
|---|---|---|---|
| Center-West | 0.357 [0.336, 0.377] | -0.184 | 0.881 |
| North | -0.046 [-0.088, -0.005] | -0.235 | 0.861 |
| Northeast | 0.041 [0.023, 0.059] | -0.010 | 0.978 |
| South | 0.085 [0.062, 0.107] | 0.101 | 0.751 |
| Southeast | 0.287 [0.268, 0.304] | -0.084 | 0.741 |

**Uncertainty.** Raw quantile intervals under-covered (0.704 temporal). After
conformal calibration, coverage was 0.831 (climate only) and 0.878 (with past
cases) on 2023 to 2025, ranging by year from 0.768 (2024) to 0.886 (2023),
and 0.741 to 0.978 across held-out regions.

**Transfer to India** (Table 3). Mean Spearman over 2015 to 2024 was 0.146
(range -0.065 to 0.310), with every yearly 95% state-bootstrap CI including 0;
population density alone gave 0.258 and the previous reported year 0.695.
For 2024 the predicted national monthly curve had Spearman 0.855 with reported
monthly cases, with the predicted peak in August and the reported peak in
October. 24.7% of district-months (442 of 641 districts) had at least one input
outside the Brazil training range, most often maximum temperature and
elevation. A post-hoc model without elevation reached mean Spearman 0.199 in
India and skill -0.014 in the Brazil temporal test.

| Table 3: ranking method | Mean Spearman (range) | Mean top-5 overlap |
|---|---|---|
| Brazil-trained climate model | 0.146 (-0.065 to 0.310) | 1.6 |
| Population density only | 0.258 (0.064 to 0.477) | 2.7 |
| Previous reported year | 0.695 (0.562 to 0.824) | 3.3 |
| Without elevation (diagnostic) | 0.199 (-0.090 to 0.407) | 1.9 |

**Explanations.** Mean 2 m temperature was the most important climate
variable at all lags (mean |SHAP| 0.090 to 0.094). Its binned SHAP curve
peaked at 25.8, 27.5 and 26.2 deg C for lags 1, 2 and 3 and declined above.
Rainfall (CHIRPS) and humidity had positive monotone associations (Spearman
of value with SHAP 0.65 to 0.69 and 0.53 to 0.89). Elevation had the largest
mean |SHAP| (0.292) with a positive association (0.83).

# 5. Discussion

**Climate explains the season, not the epidemic year.** In the temporal
hold-out, the climate-only model is almost indistinguishable from each
municipality's own seasonal climatology (skill 0.012). The test years
2023 to 2025 were far more intense than the training years (mean $y$ of 2.23,
3.27 and 2.60 against 0.98 to 2.15 in 2015 to 2022), and the climate model
raised its 2024 prediction only slightly (mean 1.61 against 1.49 for
climatology). Multi-year swings in dengue are driven largely by population
immunity and serotype replacement, which no climate covariate observes.
Adding past incidence lifts skill to 0.660, but this is mostly persistence and
assumes last month's cases are known, which reporting delays often prevent.

**Where climate helps is where history is missing.** When whole regions are
held out, the only history-free alternative is a pooled seasonal average, and
the climate model beats it in four of five regions (skill up to 0.357 in the
Center-West). It does not, however, beat a held-out place's own seasonal
history in four of five regions. The practical reading: a climate model is a
reasonable prior for places without surveillance, and should be replaced by
local data as soon as they exist.

**Explanations reveal both biology and confounding.** The learned temperature
response is unimodal with a peak between 25.8 and 27.5 deg C, matching
mechanistic estimates, and lagged rainfall and humidity act in the expected
direction. Elevation is the single most important feature, but with the wrong
sign for biology: in Brazil the high-burden cities sit on the central plateau.
This proxy is exactly what broke the India transfer, where Uttarakhand (mean
district elevation 1,813 m, beyond Brazil's 1,580 m maximum) was ranked
highest almost entirely through its elevation SHAP contribution.

**Cross-country transfer of spatial ranks failed; seasonality transferred.**
The Brazil-trained model ranked Indian states worse than population density
alone and far worse than last year's reported numbers. Yet its national
monthly curve for 2024 tracked reported cases (Spearman 0.855), peaking two
months early. Transfer should therefore be evaluated per component (timing
versus spatial ranking), and features that encode a source country's geography
(elevation, land cover mixes) should be screened before transfer, for example
by flagging inputs outside the training range, as the map now does.

# 6. Limitations

Labels are notified cases and reflect testing and reporting practice, which
differ strongly between Brazil and India and between Indian states (West
Bengal has no record for 2018 and 2019 and an implausibly low 2024 total).
Indian validation is at state-year level only; district maps are unvalidated.
Population is a single 2020 GHSL epoch for India and nearly constant per
municipality in InfoDengue. Conformal intervals assume exchangeability between
calibration and test months, which year-to-year shifts violate; empirical
coverage was below nominal on later years and varied from 0.74 to 0.98 across
held-out regions. The no-elevation analysis was chosen after seeing the India
results and is reported as a diagnostic, not as a validated improvement.

# Reproducibility

Code, configuration and the full result tables are in the repository. One
command (`python -m dengue.cli all`) runs every stage; raw pulls are cached
with pull dates; seeds are fixed (0, 1, 2); runs are logged to MLflow with the
git hash. To produce a PDF: `pandoc report/report.md -o report.pdf`.

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
