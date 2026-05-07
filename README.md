# Statistical Prediction of Rainy Season Onset in East Africa

This repository contains code, data processing pipelines, and analysis notebooks for the project:

**Statistical Prediction of Rainy Season Onset in East Africa**

The study develops interpretable statistical models to predict October–December (OND) rainy season onset across the Greater Horn of Africa using satellite and reanalysis climate data, with a focus on soil moisture, sea surface temperature (SST), and atmospheric variables.

---

## Overview

Rainy season onset in East Africa is highly variable and critical for agricultural planning. This project:

- Estimates OND onset dates using the Moron–Robertson Index
- Builds interpretable predictive models using Lasso-regularized regression and classification
- Evaluates predictors such as:
  - Sea Surface Temperature (SST)
  - Soil Moisture (surface and root-zone)
  - Precipitation (IMERG)
  - Atmospheric variables (ERA5)

Two modeling tasks are explored:

- Regression: Predict exact onset day-of-year
- Classification: Predict early / normal / late onset terciles

---

## Repository Structure
```bash
├── data/
├── notebooks/
│   ├── data_processing/
│   ├── mode_classification/
│   ├── graphs/
│   ├── regression/
│   └── classification/
├── scripts/
│   ├── classification_graphs.py
│   ├── classification.py
│   ├── correlation_graphs.py
│   ├── data_loader.py
│   ├── data_processing.py
│   ├── mode_classification.py
│   ├── onset_graphs.py
│   ├── onset_scripts.py
│   ├── regression_graphs.py
│   └── regression.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Methods

### Onset Detection

- Moron–Robertson Index
- Detects sustained wet periods while filtering false rainfall events
- Applied to IMERG precipitation (2001–2022)

### Models

#### Regression
- Lasso-regularized linear regression
- Predicts continuous onset day-of-year
- Evaluated using RMSE, R², and skill vs climatology

#### Classification
- One-vs-Rest logistic regression
- Predicts:
  - Early onset
  - Normal onset
  - Late onset
- Evaluated using accuracy, F1 score, ROC AUC

---

## Key Predictors

- Western Indian Ocean SST (Aug–Sep)
- Total Column Water Vapor (TCWV)
- Total Cloud Cover (TCC)
- Surface Soil Moisture (SMAP)
- Precipitation (IMERG)
- Latitude (strong spatial gradient)

---

## Key Results

- Best regression model:
  - RMSE ≈ 17.8 days
  - R² ≈ 0.40
  - ~31% improvement over climatology baseline

- Late onset is most predictable (ROC AUC ≈ 0.86)
- Early onset is least predictable
- Persistent spatial errors in southwestern domain

---

## Data Sources

- NASA SMAP (soil moisture)
- NASA IMERG (precipitation)
- ECMWF ERA5 (SST and atmospheric variables)

All datasets are regridded to 1° resolution and aggregated to 10-day (dekad) intervals.

---

## Requirements

```bash
xarray
pandas
numpy
xarray-regrid
regionmask
cartopy
scikit-learn
matplotlib
seaborn
scipy