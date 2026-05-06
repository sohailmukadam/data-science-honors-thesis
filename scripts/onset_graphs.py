import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from data_loader import load_onset, load_sm

def composite_onset_graph(year, days_before_onset, days_after_onset):
    """
    Create a composite time series centered on rainy season onset.

    This function:
    - Loads precipitation and soil moisture datasets.
    - Extracts time series at each grid cell for a given year.
    - Aligns all time series relative to onset date (day 0).
    - Interpolates values onto a common relative time axis.
    - Computes median and interquartile range (IQR) across grid cells.
    - Plots composite evolution of soil moisture and precipitation.

    Parameters:
    year (int): Year to analyze
    days_before_onset (int): Days before onset (negative window)
    days_after_onset (int): Days after onset (positive window)

    Input:
    $SCRATCH/onset/*.nc4
    $SCRATCH/soil_moisture/*.nc4

    Output:
    {year}_composite_onset.png

    Notes:
    - Uses grid-cell aggregation (spatial composite).
    - Requires at least 5 valid samples per time step.
    """
    precipitation = load_onset().sel(lat=slice(0, 6), lon=slice(37, 42))
    sm = load_sm().sel(lat=slice(0, 6), lon=slice(37, 42))   
    colors = {0: "#2166ac", 1: "#d6604d"}
    precip_year = precipitation.sel(time=str(year))
    sm_year = sm.sel(time=str(year))
    lat_vals = precip_year.lat.values
    lon_vals = precip_year.lon.values
    rel_days = np.arange(-1 * days_before_onset, days_after_onset + 1)
    soil_all = []
    precip_all = []
    precip_var = list(precip_year.data_vars)[0]
    sm_var = list(sm_year.data_vars)[0]
    print("Processing grid points...")
    for lat in lat_vals:
        for lon in lon_vals:
            precip_point = precip_year.sel(lat=lat, lon=lon)
            sm_point = sm_year.sel(lat=lat, lon=lon)
            onset = np.unique(precip_point.rainy_season_onset_date_h2.values)
            if np.isnat(onset):
                continue
            t_rel_precip = (precip_point.time.values - onset) / np.timedelta64(1, "D")
            t_rel_sm = (sm_point.time.values - onset) / np.timedelta64(1, "D")
            soil = sm_point[sm_var].values
            precip = precip_point[precip_var].values
            valid_soil = np.isfinite(soil)
            valid_precip = np.isfinite(precip)
            if valid_soil.sum() < 10 or valid_precip.sum() < 10:
                continue
            soil_interp = np.interp(
                rel_days,
                t_rel_sm[valid_soil],
                soil[valid_soil],
                left=np.nan,
                right=np.nan
            )
            precip_interp = np.interp(
                rel_days,
                t_rel_precip[valid_precip],
                precip[valid_precip],
                left=np.nan,
                right=np.nan
            )
            soil_all.append(soil_interp)
            precip_all.append(precip_interp)
    soil_all = np.array(soil_all)
    precip_all = np.array(precip_all)
    print(f"Number of samples in composite: {soil_all.shape[0]}")
    soil_median = np.nanmedian(soil_all, axis=0)
    soil_p25 = np.nanpercentile(soil_all, 25, axis=0)
    soil_p75 = np.nanpercentile(soil_all, 75, axis=0)
    precip_median = np.nanmedian(precip_all, axis=0)
    precip_p25 = np.nanpercentile(precip_all, 25, axis=0)
    precip_p75 = np.nanpercentile(precip_all, 75, axis=0)
    n_valid_soil = np.sum(np.isfinite(soil_all), axis=0)
    n_valid_precip = np.sum(np.isfinite(precip_all), axis=0)
    min_samples = 5
    soil_median[n_valid_soil < min_samples] = np.nan
    soil_p25[n_valid_soil < min_samples] = np.nan
    soil_p75[n_valid_soil < min_samples] = np.nan
    precip_median[n_valid_precip < min_samples] = np.nan
    precip_p25[n_valid_precip < min_samples] = np.nan
    precip_p75[n_valid_precip < min_samples] = np.nan
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(rel_days, soil_median, linewidth=2, color=colors[1], label="Soil Moisture")
    ax1.fill_between(rel_days, soil_p25, soil_p75, alpha=0.3, color=colors[1], label="Soil IQR")
    ax1.axvline(0, linestyle="--", color="k", linewidth=2, label="Onset")
    ax1.set_ylabel("Soil Moisture (m³/m³)")
    ax1.set_xlabel("Days Relative to Onset")
    ax1.legend(loc="upper left")
    ax1.set_title(f"{year} Composite: Rainy Season Onset")
    ax2 = ax1.twinx()
    ax2.plot(rel_days, precip_median, linewidth=2, color=colors[0], label="Precipitation")
    ax2.fill_between(rel_days, precip_p25, precip_p75, alpha=0.2, color=colors[0], label="Precip IQR")
    ax2.set_ylabel("Precipitation (mm/day)")
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper right")
    ax3.plot(rel_days, n_valid_soil, linewidth=2, color=colors[0], label='Soil Moisture')
    ax3.plot(rel_days, n_valid_precip, linewidth=2, color=colors[1], label='Precipitation')
    ax3.axhline(min_samples, linestyle="--", color="green", linewidth=1, label=f"Min threshold ({min_samples})")
    ax3.axvline(0, linestyle="--", color="k", linewidth=2)
    ax3.set_ylabel("Number of Valid Samples")
    ax3.set_xlabel("Days Relative to Onset")
    ax3.legend(loc="upper left")
    ax3.set_title("Sample Size Across Composite Window")
    plt.tight_layout()
    plt.savefig(f"{year}_composite_onset.png", dpi=300)
    plt.show()

def smp_time_series(seed):
    """
    Plot example time series of precipitation and soil moisture.

    This function:
    - Randomly samples grid points and years.
    - Extracts OND (Sep–Dec) time series.
    - Plots precipitation and soil moisture together.
    - Marks onset date on each plot.

    Parameters:
    seed (int): seed for plots

    Input:
    $SCRATCH/onset/*.nc4
    $SCRATCH/soil_moisture/*.nc4

    Output:
    smp_time_series.png

    Notes:
    - Uses interpolation to fill soil moisture gaps.
    - Intended for visualization, not statistical analysis.
    """
    precipitation = load_onset().sel(lat=slice(0, 6), lon=slice(37, 42))
    sm = load_sm().sel(lat=slice(0, 6), lon=slice(37, 42))   
    years = np.arange(2016, 2024)
    lats = sm.lat.values
    lons = sm.lon.values
    random.seed(seed)
    samples = [
        (random.choice(years), random.choice(lats), random.choice(lons))
        for _ in range(2)
    ]
    colors = {0: "#2166ac", 1: "#d6604d"}
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=False, sharey=False)
    axes = axes.flatten()
    twins = []
    for i, (year, lat, lon) in enumerate(samples):
        ax = axes[i]
        ax2 = ax.twinx()
        twins.append(ax2)
        precipitation_point = precipitation.sel(
            time=slice(f"{year}-09-01", f"{year}-12-31"),
            lat=lat,
            lon=lon
        )
        sm_point = sm.sel(
            time=slice(f"{year}-09-01", f"{year}-12-31"),
            lat=lat,
            lon=lon
        )
        sm_interp = sm_point.soil_moisture.interpolate_na(dim="time", method="linear")
        onset_date = precipitation_point.rainy_season_onset_date_h2.values
        ax.plot(
            precipitation_point.time,
            precipitation_point.precipitation,
            linestyle="-",
            color=colors[0],
            label="Precipitation (mm/day)"
        )
        ax.set_ylim(0, 40)
        ax2.plot(
            sm_interp.time,
            sm_interp,
            linestyle="-",
            color=colors[1],
            label="Soil Moisture (m³/m³)"
        )
        ax.set_ylabel("Precipitation", fontsize=8)
        ax2.set_ylabel("Soil Moisture", fontsize=8)
        if not np.isnat(onset_date).all():
            onset_ts = pd.Timestamp(onset_date.flat[0])
            ax.axvline(
                onset_ts,
                linestyle="--",
                linewidth=2,
                color="black",
                label="Rainy Season Onset"
            )
        ax.set_title(f"{year}, Lat {lat:.2f}, Lon {lon:.2f}", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7)
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = twins[0].get_legend_handles_labels()
    fig.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper center", ncol=3, fontsize=8,
        bbox_to_anchor=(0.5, 0.92)
    )
    fig.suptitle(
        "Precipitation & Soil Moisture Time Series (OND Rainy Season Onset)",
        fontsize=10,
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("smp_time_series.png", dpi=300)
    plt.show()