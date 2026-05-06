import glob
import xarray as xr
import os
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from data_loader import load_onset, load_sm, load_sst

def make_correlation_df():
    """
    Construct a dataframe of predictors and OND onset, and compute correlations.

    This function:
    - Loads precipitation (onset), soil moisture, and SST datasets.
    - Subsets to East Africa region.
    - Extracts dekadal (10-day) averages for selected predictors.
    - Matches predictors with OND onset timing at each grid cell and year.
    - Computes spatially-resolved Pearson correlations.
    - Estimates confidence intervals using Fisher z-transform.

    Input:
    $SCRATCH/onset/*.nc4
    $SCRATCH/soil_moisture/*.nc4
    $SCRATCH/sst/*.nc4

    Output:
    df (DataFrame): Raw dataset of predictors + onset
    correlation_df (DataFrame): Spatial correlations with confidence intervals

    Predictors:
    - aug_sm_dekad1: Surface soil moisture (Aug 1–10)
    - sep_sst_dekad1: Sea surface temperature (Sep 1–10)
    - sep_rzsm_dekad1: Root-zone soil moisture (Sep 1–10)
    - sep_precip_dekad1: Precipitation (Sep 1–10)

    Notes:
    - Correlations are computed per grid cell across years.
    - Significance is based on whether 95% CI excludes zero.
    """
    precipitation = load_onset().sel(lat=slice(0, 6), lon=slice(37, 48))
    sm = load_sm().sel(lat=slice(0, 6), lon=slice(37, 48))
    sst = load_sst().sel(lat=slice(-15, 5), lon=slice(45, 60))
    years = np.arange(2001, 2023)
    sm_lat = sm["lat"].values
    sm_lon = sm["lon"].values
    ond_onsets = []
    sep_precip_dekad1 = []
    sep_sst_dekad1 = []
    sep_rzsm_dekad1 = []
    aug_sm_dekad1 = []
    onset_years = []
    lats = []
    lons = []
    for year in years:
        dekads = {
            "aug_dekad1": (f"{year}-08-01", f"{year}-08-10"),
            "aug_dekad2": (f"{year}-08-11", f"{year}-08-20"),
            "aug_dekad3": (f"{year}-08-21", f"{year}-08-31"),
            "sep_dekad1": (f"{year}-09-01", f"{year}-09-10"),
            "sep_dekad2": (f"{year}-09-11", f"{year}-09-20"),
            "sep_dekad3": (f"{year}-09-21", f"{year}-09-30"),
        }
        for lat, lon in product(sm_lat, sm_lon):
            onset = np.unique(precipitation["rainy_season_onset_h2"].sel(time=str(year), lat=lat, lon=lon))[0]
            if np.isfinite(onset):
                aug_sm_dekad1.append(float(np.nanmean(sm["soil_moisture"].sel(time=slice(*dekads["aug_dekad1"]), lat=lat, lon=lon))))
                sep_sst_dekad1.append(float(np.nanmean(sst["SSTC"].sel(time=slice(*dekads["sep_dekad1"])))))
                sep_rzsm_dekad1.append(float(np.nanmean(sm["root_zone_soil_moisture"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                sep_precip_dekad1.append(float(np.nanmean(precipitation["precipitation"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                ond_onsets.append(onset)
                onset_years.append(year)
                lats.append(lat)
                lons.append(lon)
    df = pd.DataFrame({
        "year": onset_years, 
        "lat": lats, 
        "lon": lons,
        "ond_onset": ond_onsets,
        "aug_sm_dekad1":  aug_sm_dekad1,
        "sep_sst_dekad1": sep_sst_dekad1,
        "sep_rzsm_dekad1":sep_rzsm_dekad1,
        "sep_precip_dekad1":sep_precip_dekad1,
    })
    predictors = ["sep_sst_dekad1", "aug_sm_dekad1", "sep_rzsm_dekad1", "sep_precip_dekad1"]
    all_corrs = []
    for var in predictors:
        temp = (
            df.dropna(subset=["ond_onset", var])
            .groupby(["lat", "lon"])
            .apply(lambda g: (
                lambda r= g["ond_onset"].corr(g[var]), n=len(g): pd.Series({
                    "r": r,
                    "n": n,
                    "ci_low": np.tanh(np.arctanh(r) - 1.96 / np.sqrt(n-3)) if n>3 else np.nan,
                    "ci_high": np.tanh(np.arctanh(r) + 1.96 / np.sqrt(n-3)) if n>3 else np.nan
                })
            )())
            .reset_index()
        )
        temp["variable"] = var
        all_corrs.append(temp)
    correlation_df = pd.concat(all_corrs, ignore_index=True)
    correlation_df["significant"] = ~((correlation_df["ci_low"] < 0) & (correlation_df["ci_high"] > 0))
    return df, correlation_df

def correlation_map(correlation_df):
    """
    Plot spatial maps of correlation between predictors and OND onset.

    This function:
    - Creates a map for each predictor.
    - Displays Pearson correlation coefficient (r).
    - Overlays values where correlations are statistically significant.

    Parameters:
    correlation_df (DataFrame): Output from make_correlation_df()

    Output:
    PNG files for each predictor (saved to current directory)
    """
    predictors = ["sep_sst_dekad1", "aug_sm_dekad1", "sep_rzsm_dekad1", "sep_precip_dekad1"]
    pretty_names = {
        "sep_sst_dekad1":    "Sea Surface Temperature (Sep Dekad 1)",
        "aug_sm_dekad1":     "Surface Soil Moisture (Aug Dekad 1)",
        "sep_rzsm_dekad1":   "Root-Zone Soil Moisture (Sep Dekad 1)",
        "sep_precip_dekad1": "Precipitation (Sep Dekad 1)",
    }
    lat_full = np.arange(-1, 8)
    lon_full = np.arange(36, 50)
    proj = ccrs.PlateCarree()
    for var in predictors:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=proj)
        subset = correlation_df[correlation_df["variable"] == var].copy()
        subset.loc[subset["n"] < 4, "r"] = np.nan
        pivot  = (
            subset.pivot(index="lat", columns="lon", values="r")
                .reindex(index=lat_full, columns=lon_full)
        )
        R    = pivot.values
        lats = pivot.index.values
        lons = pivot.columns.values
        lon2d, lat2d = np.meshgrid(lons, lats)
        sig_pivot = (
            subset.pivot(index="lat", columns="lon", values="significant")
                .reindex(index=lat_full, columns=lon_full)
                .fillna(False)
                .astype(bool)
        )
        sig = sig_pivot.values
        pcm = ax.pcolormesh(
            lon2d, lat2d, R,
            cmap="RdBu_r", vmin=-1, vmax=1,
            shading="auto", transform=proj
        )
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                val = R[i, j]
                if np.isnan(val):
                    continue
                if not sig[i, j]:
                    continue
                ax.text(
                    lon, lat, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=6,
                    color="black",
                    transform=proj,
                )
        ax.coastlines(resolution="50m")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND,  facecolor="lightgrey", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
        ax.set_extent([lon_full[0]-0.5, lon_full[-1]+0.5, lat_full[0]-0.5, lat_full[-1]+0.5], crs=proj)
        ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="grey")
        plt.colorbar(pcm, ax=ax, label="Pearson r", shrink=0.85)
        ax.set_title(f"Correlation with OND Onset — {pretty_names[var]}", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{var}_correlation.png", dpi=300, bbox_inches="tight")
        plt.show()

def correlation_graph(df):
    """
    Plot spatial maps of correlation between predictors and OND onset.

    This function:
    - Creates a map for each predictor.
    - Displays Pearson correlation coefficient (r).
    - Overlays values where correlations are statistically significant.

    Parameters:
    correlation_df (DataFrame): Output from make_correlation_df()

    Output:
    PNG files for each predictor (saved to current directory)
    """
    predictors = ["sep_sst_dekad1", "aug_sm_dekad1", "sep_rzsm_dekad1", "sep_precip_dekad1"]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5))
    axes = axes.flatten()
    df_2016 = df[df["year"] >= 2016]
    for ax, (col, xlabel) in zip(axes, predictors):
        x, y = df_2016[col].values, df_2016["ond_onset"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        n = len(x)
        ax.scatter(x, y, s=18, color="#3a6ea5", alpha=0.65, linewidths=0, zorder=3)
        r, _ = stats.pearsonr(x, y)
        m, b, *_ = stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 200)
        ax.plot(xl, m * xl + b, color="#999", linewidth=0.9, linestyle="--", zorder=2)
        if n > 3:
            ci_low  = np.tanh(np.arctanh(r) - 1.96 / np.sqrt(n - 3))
            ci_high = np.tanh(np.arctanh(r) + 1.96 / np.sqrt(n - 3))
            sig = not (ci_low <= 0 <= ci_high)
        else:
            sig = False
        label = f"$r$ = {r:.2f}{'*' if sig else ''}"
        ax.text(0.05, 0.95, label,
                transform=ax.transAxes, ha="left", va="top", fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("OND onset", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.6)
        ax.tick_params(width=0.6)
    plt.tight_layout(pad=1.2, h_pad=1.8, w_pad=1.2)
    plt.savefig("ond_onset_scatter.png", dpi=300, bbox_inches="tight")