import numpy as np
from scipy.signal import welch
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from data_loader import load_onset

def classify_from_harmonics(ts, variance_threshold=0.10):
    """
    Classify a precipitation time series based on harmonic (frequency) content.

    This function:
    - Removes the mean to compute anomalies.
    - Estimates the power spectral density (PSD) using Welch’s method.
    - Computes the fraction of variance explained by annual and semi-annual cycles.
    - Classifies the signal based on dominant periodicity.

    Parameters:
    ts (array-like): Time series (e.g., monthly precipitation values)
    variance_threshold (float): Minimum spectral dominance required for classification

    Returns:
    label (str): One of:
        - "unimodal": annual cycle dominant
        - "bimodal": semi-annual cycle dominant
        - "none": no clear periodic structure or invalid data
    mode_id (int): Encoded label (0=none, 1=unimodal, 2=bimodal)

    Notes:
    - Assumes monthly data with fs=12 (12 samples per year).
    - Uses frequency bins near 1 (annual) and 2 (semi-annual).
    """
    if np.any(np.isnan(ts)) or ts.max() == 0:
        return "ocean", 0
    ts_anom = ts - ts.mean()
    total_var = np.var(ts_anom)
    if total_var == 0:
        return "none", 0
    freqs, psd = welch(ts_anom, fs=12, nperseg=12)
    total_psd = psd.sum()
    if total_psd == 0:
        return "none", 0
    psd_frac = psd / total_psd
    annual_idx = np.argmin(np.abs(freqs - 1))
    semiannual_idx = np.argmin(np.abs(freqs - 2))
    annual_frac = psd_frac[annual_idx]
    semiannual_frac = psd_frac[semiannual_idx]
    dominant = max(annual_frac, semiannual_frac)
    if dominant < variance_threshold:
        return "none", 0
    elif semiannual_frac > annual_frac:
        return "bimodal", 2
    else:
        return "unimodal", 1
    
def classify_mode(save_data=False, plot=False, save_plot=False, variance_threshold=0.10):
    """
    Classify spatial precipitation regimes using harmonic analysis.

    This function:
    - Loads ERA5 precipitation data across multiple years.
    - Computes monthly climatology.
    - Applies harmonic classification at each grid cell.
    - Produces a spatial map of precipitation regimes.
    - Optionally saves results and/or plots classification map.

    Parameters:
    save_data (bool): Save output dataset to NetCDF
    plot (bool): Display classification map
    save_plot (bool): Save plot as PNG
    variance_threshold (float): Threshold for spectral dominance

    Input:
    $SCRATCH/onset/ONSET_{year}.nc4 (2001–2022)

    Output (optional):
    $SCRATCH/mode/smp.nc4

    Classification:
    0 = none (no clear cycle)
    1 = unimodal (annual cycle dominant)
    2 = bimodal (semi-annual cycle dominant)
    """
    ds = load_onset()
    monthly_clim = ds["precipitation"].resample(time="1MS").mean()
    lats = ds.lat.values
    lons = ds.lon.values
    mode_map = np.empty((len(lats), len(lons)), dtype=object)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            ts = monthly_clim.sel(lat=lat, lon=lon).values
            label, _ = classify_from_harmonics(ts, variance_threshold)
            mode_map[i, j] = label
    mode_int = np.where(mode_map == "bimodal", 2, np.where(mode_map == "unimodal", 1, 0))
    mode_da = xr.DataArray(
        mode_int,
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
        name="mode"
    )
    ds["mode"] = mode_da
    if save_data:
        output_dir = os.path.expandvars("$SCRATCH/mode")
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/smp.nc4"
        if os.path.isfile(output_file):
            os.remove(output_file)
        ds.to_netcdf(output_file, mode="w", format="NETCDF4")
    if plot:
        colors = {0: "#ffffff", 1: "#2166ac", 2: "#d6604d"}
        cmap = mcolors.ListedColormap([colors[k] for k in sorted(colors)])
        fig, ax = plt.subplots(figsize=(7, 6))
        ds["mode"].plot(ax=ax, cmap=cmap, add_colorbar=False)
        legend_elements = [
            mpatches.Patch(facecolor=colors[1], label="Unimodal"),
            mpatches.Patch(facecolor=colors[2], label="Bimodal"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", title="Precipitation Regime")
        ax.set_title("Precipitation Mode Classification")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        rect = Rectangle((36.5, -0.5), 12, 7, linewidth=2, edgecolor="black", facecolor="none")
        ax.add_patch(rect)
        ax.text(40, 2.7, "Prediction Region", fontsize=10)
        if save_plot:
            plt.savefig("mode_classification.png", dpi=300, bbox_inches="tight")
        plt.show()