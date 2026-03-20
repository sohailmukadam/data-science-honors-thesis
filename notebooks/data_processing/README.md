# Data Processing

## Overview

This directory contains the data processing pipeline for generating gridded climate variables and rainy season onset metrics over East Africa. The workflow integrates multiple datasets (SMAP, IMERG, ERA5) and standardizes them into a consistent spatial (1° × 1°) and temporal (daily) resolution.

The pipeline is designed to:

* Ingest raw climate datasets.
* Subset to the East Africa region.
* Regrid to a common spatial grid.
* Compute derived variables (e.g., rainy season onset).
* Save outputs in NetCDF format for downstream analysis.

---

## Directory Structure

```
scripts/
│── data_processing.py        # Main data processing functions
│── onset.py                  # Rainy season onset detection algorithm
```
```
notebooks/data_processing/
│── era5.ipynb
│── soil_moisture.ipynb
│── precipitation.ipynb
│── sea_surface_temperature.ipynb
│── README.md                 # Documentation (this file)
```

### Data

All data is read from and written to the `$SCRATCH` directory:

```
$SCRATCH/
│── SMAP_2016_2023.nc
│── soil_moisture/
│   └── SMAP_{year}.nc4
│── precipitation/
│   └── IMERG_{year}.nc4
│── onset/
│   └── ONSET_{year}.nc4
│── sst/
│   └── ERA5_{year}.nc4
│── era5/
│   └── ERA5_{year}.nc4
```

Raw input datasets are in external directories (e.g., IMERG, ERA5, SMAP).

---

## Datasets

### 1. SMAP Soil Moisture

* **Source:** ...
* **Variables:**

    * `surface_sm` → surface soil moisture
    * `root_zone_sm` → root zone soil moisture

* **Processing:**
    * process_smap_data(year)
    * Extract year
    * Convert to time-indexed format
    * Regrid to 1° grid

* **Output:**

```
$SCRATCH/soil_moisture/SMAP_{year}.nc4
```

---

### 2. IMERG Precipitation

* **Source:** ...
* **Variables:**

    * `precipitation` or `precipitationCal`

* **Processing:**

    * process_precipitation_data(year)
    * Merge daily files
    * Subset to East Africa
    * Regrid to 1° grid
    * Apply land mask

* **Output:**

```
$SCRATCH/precipitation/IMERG_{year}.nc4
```

---

### 3. Rainy Season Onset

* **Derived from:** IMERG precipitation
* **Method:** rainy_season_onset_detection (`onset_scripts.py`)
* **Outputs:**

    * `rainy_season_onset` (DOY)
    * `rainy_season_onset_h2` (DOY, October-December season)
    * Corresponding datetime variables
* **Output:**

```
$SCRATCH/onset/ONSET_{year}.nc4
```

---

### 4. ERA5 Sea Surface Temperature (SST)

* **Variables:**

    * `SSTK` (Kelvin)
    * `SSTC` (Celsius, derived)
* **Processing:**
    * process_sst_data(year)
    * Hourly → daily mean
    * Subset to Western Indian Ocean
    * Regrid to 1° grid
* **Output:**

```
$SCRATCH/sst/ERA5_{year}.nc4
```

---

### 5. ERA5 Atmospheric Variables

* **Variable Examples:**

    * `2t` (2-meter temperature)
    * `sp` (surface pressure)
    * `tcwv` (total column water vapor)
* **Processing:**
    * process_era5_data(year, variables)
    * Merge variables
    * Hourly → daily mean
    * Regrid to 1° grid
* **Output:**

```
$SCRATCH/era5/ERA5_{year}.nc4
```

---

## Spatial Domain

### East Africa Region

* Latitude: **[-10, 15]**
* Longitude: **[30, 52]**

### SST Region (Western Indian Ocean)

* Latitude: **[-15, 5]**
* Longitude: **[45, 60]**

---

## Workflow

### Step 1: Process Raw Data

```python
process_smap_data(year)
process_precipitation_data(year)
process_sst_data(year)
process_era5_data(year, variables)
```

### Step 2: Compute Derived Variables

```python
modify_precipitation_data(year)
```

---

## Rainy Season Onset Algorithm

Implemented in `onset_scripts.py`.

### Key Steps:

1. Classify wet vs dry days
2. Compute rolling wet and dry windows
3. Identify candidate onset (O1)
4. Confirm onset (O2) if no dry spell follows

### Key Parameters:

* Wet window length: 3 days
* Wet threshold: 20 mm
* Dry window length: 7 days
* Confirmation window: 20 days

---

## Dependencies

* xarray
* numpy
* pandas
* xarray_regrid
* regionmask
* glob
* os

---

## Notes & Assumptions

* All datasets are regridded to **1° × 1° resolution**.
* Time is standardized to **daily resolution**.
* Missing values are not explicitly handled in onset detection.
* Leap years are treated as 365-day years in onset calculations.
* Input file naming conventions must match expected patterns.

---

## Reproducibility

* All outputs are written as **NetCDF4 files**.
* Intermediate data is stored in `$SCRATCH`.
* Processing is **year-by-year*.
* Raw data is not modified; only derived outputs are stored.

---
