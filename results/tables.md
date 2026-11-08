### Temporal hold-out (train 2015 to 2021, calibrate 2022, test 2023 to 2025)

Metrics on log1p(monthly incidence per 100k). Brackets: 95% municipality cluster bootstrap CI (1,000 resamples). LightGBM rows use the mean of 3 seeds.

| model | MAE | RMSE | R2 | skill vs climatology | Spearman | alert F1 | alert AUC-PR | n |
|---|---|---|---|---|---|---|---|---|
| climatology | 1.758 [1.736, 1.781] | 2.367 [2.345, 2.392] | 0.104 [0.087, 0.119] | 0.000 [-0.000, 0.000] | 0.576 | 0.415 | 0.515 | 200520 |
| last_year | 1.637 [1.621, 1.656] | 2.414 [2.394, 2.436] | 0.069 [0.050, 0.085] | -0.040 [-0.060, -0.021] | 0.540 | 0.612 | 0.627 | 200520 |
| lgbm_climate (3-seed mean) | 1.821 [1.799, 1.843] | 2.354 [2.331, 2.377] | 0.115 [0.099, 0.130] | 0.012 [0.006, 0.017] | 0.602 | 0.396 | 0.520 | 200520 |
| lgbm_ar (3-seed mean) | 0.984 [0.974, 0.995] | 1.380 [1.368, 1.391] | 0.696 [0.690, 0.701] | 0.660 [0.654, 0.667] | 0.821 | 0.679 | 0.806 | 200520 |

Seed spread (single-seed models):

| model | mae_mean | mae_sd | skill_mean | skill_sd |
|---|---|---|---|---|
| lgbm_ar | 0.9862 | 0.0003 | 0.6591 | 0.0003 |
| lgbm_climate | 1.8223 | 0.001 | 0.0096 | 0.0009 |

By test year:

| year | model | mae | rmse | skill_vs_clim |
|---|---|---|---|---|
| 2023 | clim | 1.447 | 1.992 | 0.0 |
| 2023 | pred_lgbm_climate | 1.58 | 2.069 | -0.08 |
| 2023 | pred_lgbm_ar | 0.943 | 1.331 | 0.553 |
| 2024 | clim | 2.162 | 2.837 | 0.0 |
| 2024 | pred_lgbm_climate | 2.125 | 2.713 | 0.086 |
| 2024 | pred_lgbm_ar | 1.052 | 1.464 | 0.734 |
| 2025 | clim | 1.664 | 2.19 | 0.0 |
| 2025 | pred_lgbm_climate | 1.758 | 2.231 | -0.037 |
| 2025 | pred_lgbm_ar | 0.958 | 1.34 | 0.625 |

90% interval coverage on the test years (mean over 3 seeds):

| model | nominal | raw_coverage | raw_width | conformal_margin | conformal_coverage | conformal_width |
|---|---|---|---|---|---|---|
| lgbm_ar | 0.9 | 0.825 | 3.533 | 0.128 | 0.878 | 3.789 |
| lgbm_climate | 0.9 | 0.704 | 3.934 | 0.856 | 0.831 | 5.647 |

### Spatial blocked CV (hold out one macro-region, train on the others; test years 2023 to 2025)

| held-out region | model | MAE | skill vs pooled climatology | Spearman | alert AUC-PR | skill vs local climatology |
|---|---|---|---|---|---|---|
| Center-West | pooled_climatology | 2.771 [2.700, 2.841] | 0.000 [-0.000, 0.000] | 0.500 | 0.577 |  |
| Center-West | local_climatology (uses own history) | 1.865 [1.807, 1.923] | 0.457 [0.423, 0.486] | 0.550 | 0.636 |  |
| Center-West | lgbm_climate (3-seed mean) | 2.171 [2.112, 2.230] | 0.357 [0.336, 0.377] | 0.515 | 0.610 | -0.184 |
| North | pooled_climatology | 1.866 [1.803, 1.933] | 0.000 [-0.000, 0.000] | 0.313 | 0.440 |  |
| North | local_climatology (uses own history) | 1.502 [1.436, 1.571] | 0.153 [0.094, 0.212] | 0.517 | 0.474 |  |
| North | lgbm_climate (3-seed mean) | 1.813 [1.743, 1.883] | -0.046 [-0.088, -0.005] | 0.358 | 0.435 | -0.235 |
| Northeast | pooled_climatology | 1.621 [1.602, 1.638] | 0.000 [-0.000, 0.000] | 0.282 | 0.360 |  |
| Northeast | local_climatology (uses own history) | 1.315 [1.289, 1.341] | 0.051 [0.027, 0.075] | 0.474 | 0.373 |  |
| Northeast | lgbm_climate (3-seed mean) | 1.524 [1.504, 1.544] | 0.041 [0.023, 0.059] | 0.374 | 0.378 | -0.010 |
| South | pooled_climatology | 2.216 [2.165, 2.269] | 0.000 [-0.000, 0.000] | 0.341 | 0.497 |  |
| South | local_climatology (uses own history) | 1.845 [1.783, 1.905] | -0.018 [-0.063, 0.024] | 0.636 | 0.518 |  |
| South | lgbm_climate (3-seed mean) | 1.931 [1.878, 1.989] | 0.085 [0.062, 0.107] | 0.600 | 0.574 | 0.101 |
| Southeast | pooled_climatology | 2.948 [2.904, 2.990] | 0.000 [-0.000, 0.000] | 0.494 | 0.620 |  |
| Southeast | local_climatology (uses own history) | 2.211 [2.179, 2.246] | 0.342 [0.322, 0.360] | 0.589 | 0.692 |  |
| Southeast | lgbm_climate (3-seed mean) | 2.374 [2.336, 2.414] | 0.287 [0.268, 0.304] | 0.539 | 0.676 | -0.084 |

90% interval coverage in the held-out region:

| region | conformal_coverage | raw_coverage | conformal_width |
|---|---|---|---|
| Center-West | 0.881 | 0.717 | 6.777 |
| North | 0.861 | 0.716 | 5.52 |
| Northeast | 0.978 | 0.91 | 6.756 |
| South | 0.751 | 0.651 | 4.859 |
| Southeast | 0.741 | 0.547 | 6.181 |

### Does climate add skill beyond local seasonal climatology? (temporal test years)

| model (no recent cases) | MAE | skill vs climatology | Spearman | alert AUC-PR |
|---|---|---|---|---|
| climatology (no model) | 1.758 [1.736, 1.781] | 0.000 [-0.000, 0.000] | 0.576 | 0.515 |
| climatology_only (3-seed mean) | 1.758 [1.736, 1.781] | -0.000 [-0.000, -0.000] | 0.576 | 0.522 |
| climatology_plus_static (3-seed mean) | 1.759 [1.737, 1.782] | 0.000 [0.000, 0.000] | 0.574 | 0.522 |
| climatology_plus_climate (3-seed mean) | 1.770 [1.749, 1.793] | -0.008 [-0.010, -0.006] | 0.564 | 0.527 |

### SHAP

Mean abs SHAP by climate lag, lgbm_climate:

| variable | 1 | 2 | 3 |
|---|---|---|---|
| lst_day | 0.0331 | 0.0218 | 0.0757 |
| precip_chirps | 0.0469 | 0.0652 | 0.0496 |
| precip_era5 | 0.0369 | 0.0526 | 0.0377 |
| rh_mean | 0.047 | 0.0324 | 0.058 |
| t2m_max | 0.0606 | 0.0522 | 0.0526 |
| t2m_mean | 0.0941 | 0.0896 | 0.0944 |
| t2m_min | 0.0606 | 0.069 | 0.0778 |

Biological sanity checks, lgbm_climate:

| feature | check | peak_x | shap_at_min_x | shap_at_peak | shap_at_max_x | spearman_x_shap |
|---|---|---|---|---|---|---|
| t2m_mean_lag1 | temperature optimum | 25.828 | -0.211 | 0.121 | 0.06 | 0.746 |
| t2m_mean_lag2 | temperature optimum | 27.541 | -0.212 | 0.116 | 0.101 | 0.915 |
| t2m_mean_lag3 | temperature optimum | 26.158 | -0.204 | 0.112 | 0.012 | 0.788 |
| lst_day_lag1 | temperature optimum | 32.976 | -0.058 | 0.039 | 0.03 | 0.832 |
| lst_day_lag2 | temperature optimum | 33.187 | -0.017 | 0.021 | 0.015 | 0.702 |
| lst_day_lag3 | temperature optimum | 29.693 | -0.108 | 0.074 | 0.052 | 0.787 |
| rh_mean_lag1 | direction | nan | nan | nan | nan | 0.528 |
| rh_mean_lag2 | direction | nan | nan | nan | nan | 0.835 |
| rh_mean_lag3 | direction | nan | nan | nan | nan | 0.888 |
| precip_chirps_lag1 | direction | nan | nan | nan | nan | 0.688 |
| precip_chirps_lag2 | direction | nan | nan | nan | nan | 0.686 |
| precip_chirps_lag3 | direction | nan | nan | nan | nan | 0.652 |
| elevation | direction | nan | nan | nan | nan | 0.83 |

Mean abs SHAP by climate lag, lgbm_ar:

| variable | 1 | 2 | 3 |
|---|---|---|---|
| lst_day | 0.0344 | 0.0127 | 0.0247 |
| precip_chirps | 0.0599 | 0.0156 | 0.0112 |
| precip_era5 | 0.0445 | 0.0182 | 0.0113 |
| rh_mean | 0.0265 | 0.0132 | 0.0163 |
| t2m_max | 0.0281 | 0.0266 | 0.017 |
| t2m_mean | 0.0219 | 0.0147 | 0.0133 |
| t2m_min | 0.0513 | 0.0246 | 0.0232 |

Biological sanity checks, lgbm_ar:

| feature | check | peak_x | shap_at_min_x | shap_at_peak | shap_at_max_x | spearman_x_shap |
|---|---|---|---|---|---|---|
| t2m_mean_lag1 | temperature optimum | 25.828 | -0.06 | 0.021 | -0.01 | 0.525 |
| t2m_mean_lag2 | temperature optimum | 29.46 | 0.006 | 0.027 | 0.027 | 0.424 |
| t2m_mean_lag3 | temperature optimum | 28.276 | -0.036 | 0.021 | 0.001 | 0.468 |
| lst_day_lag1 | temperature optimum | 32.976 | -0.132 | 0.032 | 0.028 | 0.886 |
| lst_day_lag2 | temperature optimum | 31.956 | -0.011 | 0.009 | 0.005 | 0.552 |
| lst_day_lag3 | temperature optimum | 37.407 | -0.022 | 0.056 | 0.056 | 0.58 |
| rh_mean_lag1 | direction | nan | nan | nan | nan | 0.734 |
| rh_mean_lag2 | direction | nan | nan | nan | nan | 0.347 |
| rh_mean_lag3 | direction | nan | nan | nan | nan | -0.061 |
| precip_chirps_lag1 | direction | nan | nan | nan | nan | 0.915 |
| precip_chirps_lag2 | direction | nan | nan | nan | nan | 0.589 |
| precip_chirps_lag3 | direction | nan | nan | nan | nan | -0.048 |
| elevation | direction | nan | nan | nan | nan | 0.802 |

### Brazil to India transfer (state-year ranks vs reported cases)

| method | years | mean_spearman | median_spearman | min_spearman | max_spearman | mean_top5_overlap |
|---|---|---|---|---|---|---|
| model (Brazil-trained, climate only) | 10 | 0.146 | 0.154 | -0.065 | 0.31 | 1.6 |
| population density only | 10 | 0.258 | 0.257 | 0.064 | 0.477 | 2.7 |
| previous year reported (persistence) | 9 | 0.695 | 0.69 | 0.562 | 0.824 | 3.333 |

By year (Spearman with 95% state bootstrap CI):

| year | method | n_states | spearman | spearman_lo | spearman_hi | top5_overlap |
|---|---|---|---|---|---|---|
| 2015 | model (Brazil-trained, climate only) | 35 | 0.287 | -0.102 | 0.605 | 2 |
| 2015 | population density only | 35 | 0.347 | -0.059 | 0.665 | 3 |
| 2016 | model (Brazil-trained, climate only) | 35 | 0.059 | -0.361 | 0.427 | 2 |
| 2016 | population density only | 35 | 0.308 | -0.079 | 0.626 | 3 |
| 2016 | previous year reported (persistence) | 35 | 0.675 | 0.358 | 0.886 | 3 |
| 2017 | model (Brazil-trained, climate only) | 35 | 0.31 | -0.062 | 0.648 | 3 |
| 2017 | population density only | 35 | 0.428 | 0.039 | 0.734 | 3 |
| 2017 | previous year reported (persistence) | 35 | 0.756 | 0.497 | 0.895 | 4 |
| 2018 | model (Brazil-trained, climate only) | 34 | 0.141 | -0.265 | 0.513 | 1 |
| 2018 | population density only | 34 | 0.189 | -0.215 | 0.538 | 2 |
| 2018 | previous year reported (persistence) | 34 | 0.713 | 0.451 | 0.883 | 4 |
| 2019 | model (Brazil-trained, climate only) | 34 | 0.168 | -0.233 | 0.556 | 1 |
| 2019 | population density only | 34 | 0.206 | -0.163 | 0.531 | 2 |
| 2019 | previous year reported (persistence) | 34 | 0.655 | 0.348 | 0.863 | 3 |
| 2020 | model (Brazil-trained, climate only) | 35 | 0.276 | -0.121 | 0.61 | 2 |
| 2020 | population density only | 35 | 0.477 | 0.131 | 0.719 | 3 |
| 2020 | previous year reported (persistence) | 34 | 0.788 | 0.529 | 0.915 | 3 |
| 2021 | model (Brazil-trained, climate only) | 35 | 0.199 | -0.223 | 0.552 | 3 |
| 2021 | population density only | 35 | 0.327 | -0.106 | 0.642 | 4 |
| 2021 | previous year reported (persistence) | 35 | 0.824 | 0.614 | 0.923 | 4 |
| 2022 | model (Brazil-trained, climate only) | 35 | 0.059 | -0.32 | 0.409 | 1 |
| 2022 | population density only | 35 | 0.095 | -0.338 | 0.472 | 3 |
| 2022 | previous year reported (persistence) | 35 | 0.562 | 0.269 | 0.77 | 3 |
| 2023 | model (Brazil-trained, climate only) | 35 | -0.065 | -0.428 | 0.313 | 0 |
| 2023 | population density only | 35 | 0.064 | -0.325 | 0.441 | 2 |
| 2023 | previous year reported (persistence) | 35 | 0.69 | 0.401 | 0.88 | 3 |
| 2024 | model (Brazil-trained, climate only) | 35 | 0.025 | -0.327 | 0.411 | 1 |
| 2024 | population density only | 35 | 0.143 | -0.243 | 0.504 | 2 |
| 2024 | previous year reported (persistence) | 35 | 0.589 | 0.246 | 0.838 | 3 |

Seasonality check vs India national monthly cases:

| year | months | spearman_monthly | obs_peak_month | pred_peak_month |
|---|---|---|---|---|
| 2024.0 | 11.0 | 0.855 | 10.0 | 8.0 |

### Post-hoc diagnostic: model without elevation (chosen after the India results)

Brazil temporal test:

| model | mae | rmse | r2 | spearman | skill_vs_clim |
|---|---|---|---|---|---|
| lgbm_climate without elevation (3-seed mean) | 1.845 | 2.384 | 0.092 | 0.591 | -0.014 |

India state ranks:

| method | years | mean_spearman | min_spearman | max_spearman | mean_top5_overlap |
|---|---|---|---|---|---|
| model without elevation (diagnostic) | 10 | 0.199 | -0.09 | 0.407 | 1.9 |

| year | method | n_states | spearman | spearman_lo | spearman_hi | top5_overlap |
|---|---|---|---|---|---|---|
| 2015 | model without elevation (diagnostic) | 35 | 0.383 | 0.008 | 0.699 | 3 |
| 2016 | model without elevation (diagnostic) | 35 | 0.259 | -0.145 | 0.6 | 2 |
| 2017 | model without elevation (diagnostic) | 35 | 0.343 | -0.057 | 0.679 | 2 |
| 2018 | model without elevation (diagnostic) | 34 | 0.138 | -0.285 | 0.523 | 2 |
| 2019 | model without elevation (diagnostic) | 34 | 0.179 | -0.199 | 0.537 | 1 |
| 2020 | model without elevation (diagnostic) | 35 | 0.407 | 0.068 | 0.683 | 2 |
| 2021 | model without elevation (diagnostic) | 35 | 0.316 | -0.074 | 0.658 | 3 |
| 2022 | model without elevation (diagnostic) | 35 | 0.133 | -0.283 | 0.477 | 2 |
| 2023 | model without elevation (diagnostic) | 35 | -0.076 | -0.438 | 0.278 | 1 |
| 2024 | model without elevation (diagnostic) | 35 | -0.09 | -0.481 | 0.289 | 1 |
