import os
import xarray as xr
import pandas as pd
import numpy as np
import xarray_regrid
import regionmask
import glob
from onset_scripts import detect_rainy_season_onset

def process_smap_data(year):
    """
    Process SMAP soil moisture data for a given year.

    This function:
    - Loads SMAP data from a multi-year NetCDF file.
    - Extracts a single year of data.
    - Converts it into a structured format (time, lat, lon).
    - Subsets to East Africa region.
    - Regrids data to a 1° x 1° grid.
    - Saves output as a NetCDF file.

    Parameters:
    year (int): Year to process

    Input:
    $SCRATCH/SMAP_2016_2023.nc

    Output:
    $SCRATCH/soil_moisture/SMAP_{year}.nc4

    Variables:
    soil_moisture: surface soil moisture
    root_zone_soil_moisture: deeper soil moisture

    Notes:
    - Assumes input dataset contains dimensions: lat, lon, day, year.
    - Uses mean aggregation during regridding.
    """
    input_dir = os.path.expandvars("$SCRATCH")
    output_dir = os.path.expandvars("$SCRATCH/soil_moisture")
    os.makedirs(output_dir, exist_ok=True)
    sm = xr.open_dataset(f"{input_dir}/SMAP_2016_2023.nc")
    sm = sm.sel(year=year)
    soil_moisture = sm["surface_sm"].values    
    root_zone_soil_moisture = sm["root_zone_sm"].values 
    lat = sm["lat"].values                    
    lon = sm["lon"].values                   
    day_vals = sm["day"].values
    n_lat_idx, n_lon_idx, n_day = soil_moisture.shape
    lat_full = np.broadcast_to(lat[:, :, np.newaxis], (n_lat_idx, n_lon_idx, n_day))
    lon_full = np.broadcast_to(lon[:, :, np.newaxis], (n_lat_idx, n_lon_idx, n_day))
    day_full = np.broadcast_to(day_vals[np.newaxis, np.newaxis, :], (n_lat_idx, n_lon_idx, n_day))
    df = pd.DataFrame({
        "lat": lat_full.ravel(),
        "lon": lon_full.ravel(),
        "soil_moisture": soil_moisture.ravel(),
        "root_zone_soil_moisture": root_zone_soil_moisture.ravel(),
        "day": day_full.ravel()
    })
    df["time"] = pd.to_datetime(
        f"{year}" + df["day"].astype(str).str.zfill(3),
        format="%Y%j"
    )
    df = df.drop(columns=["day"])
    df = df.set_index(["time", "lat", "lon"])
    lat_min, lat_max = -10, 15
    lon_min, lon_max = 30, 52
    ds_sm = xr.Dataset.from_dataframe(df).sortby("time")
    ds_east_africa = ds_sm.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )
    target = xarray_regrid.Grid(
        north=lat_max,
        south=lat_min,
        east=lon_max,
        west=lon_min,
        resolution_lat=1,
        resolution_lon=1,
    ).create_regridding_dataset(lat_name="lat", lon_name="lon")
    da_sm = ds_east_africa["soil_moisture"].regrid.stat(
        target,
        method="mean",
        time_dim="time",
        skipna=False
    )
    da_rzsm = ds_east_africa["root_zone_soil_moisture"].regrid.stat(
        target,
        method="mean",
        time_dim="time",
        skipna=False
    )
    ds_regridded = xr.Dataset({
        "soil_moisture": da_sm,
        "root_zone_soil_moisture": da_rzsm
    })
    output_file = f"{output_dir}/SMAP_{year}.nc4"
    ds_regridded.to_netcdf(output_file, mode="w", format="NETCDF4")
    print(f"Saved {output_file}")

def process_precipitation_data(year):
    """
    Process IMERG daily precipitation data for a given year.

    This function:
    - Loads daily IMERG precipitation files for a given year.
    - Merges them into a single dataset.
    - Subsets to East Africa (lat: -10 to 15, lon: 30 to 52).
    - Regrids data to a 1° x 1° grid using mean aggregation.
    - Applies a land mask to remove ocean grid points.
    - Saves output as a NetCDF file.

    Parameters:
    year (int): Year to process

    Input:
    */IMERG_daily/*.nc4

    Output:
    $SCRATCH/precipitation/IMERG_{year}.nc4

    Notes:
    - Supports both 'precipitation' and 'precipitationCal' variable names.
    - Uses Natural Earth land mask via regionmask.
    - Assumes daily time resolution.
    """
    path = "*/IMERG_daily"
    output_dir = os.path.expandvars("$SCRATCH/precipitation")
    os.makedirs(output_dir, exist_ok=True)
    lat_min, lat_max = -10, 15
    lon_min, lon_max = 30, 52
    target = xarray_regrid.Grid(
        north=lat_max,
        south=lat_min,
        east=lon_max,
        west=lon_min,
        resolution_lat=1,
        resolution_lon=1,
    ).create_regridding_dataset(lat_name="lat", lon_name="lon")
    files = sorted(glob.glob(f"{path}/*.{year}*.nc4"))
    if not files:
        print(f"No files found for {year}")
        return
    with xr.open_mfdataset(files, combine="by_coords") as ds_p:
        ds_p = ds_p.sortby("time")
        if "precipitation" in ds_p:
            precip_var = "precipitation"
        elif "precipitationCal" in ds_p:
            precip_var = "precipitationCal"
        else:
            raise KeyError(f"No recognized precipitation variable found. Available: {list(ds_p.data_vars)}")
        print(f"Using variable: {precip_var}")
        da_east_africa_p = ds_p[precip_var].sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
        da_east_africa_p_regridded = da_east_africa_p.regrid.stat(
            target,
            method="mean",
            time_dim="time",
            skipna=False
        )
        da_east_africa_p_regridded.name = "precipitation"
        land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(da_east_africa_p_regridded)
        da_east_africa_p_regridded_masked = da_east_africa_p_regridded.where(land_mask.notnull())
        output_file = f"{output_dir}/IMERG_{year}.nc4"
        da_east_africa_p_regridded_masked.to_netcdf(output_file, mode="w", format="NETCDF4")
        print(f"Saved {output_file}")

def modify_precipitation_data(year):
    """
    Compute rainy season onset dates from processed IMERG precipitation data.

    This function:
    - Loads regridded precipitation data.
    - Reshapes data into (time, space) format.
    - Applies onset detection algorithm.
    - Computes onset for: (1) Full year and (2) second rainy season (starting Sep 16).
    - Converts onset day-of-year to calendar dates.
    - Saves results as a NetCDF file.

    Parameters:
    year (int): Year to process

    Input:
    $SCRATCH/precipitation/IMERG_{year}.nc4

    Output:
    $SCRATCH/onset/ONSET_{year}.nc4

    Outputs include:
    - rainy_season_onset (DOY)
    - rainy_season_onset_h2 (DOY)
    - rainy_season_onset_date (datetime)
    - rainy_season_onset_date_h2 (datetime)

    Notes:
    - Uses detect_rainy_season_onset() function.
    - Assumes 365-day year.
    """
    input_dir = os.path.expandvars("$SCRATCH/precipitation")
    output_dir = os.path.expandvars("$SCRATCH/onset")
    os.makedirs(output_dir, exist_ok=True)
    ds = xr.open_dataset(f"{input_dir}/IMERG_{year}.nc4")
    ds = xr.decode_cf(ds)
    ds["time"] = ds.indexes["time"].to_datetimeindex()
    da = ds["precipitation"]
    n_days = da.sizes["time"]
    spatial_dims = [d for d in da.dims if d != "time"]
    spatial_shape = tuple(da.sizes[d] for d in spatial_dims)
    X = da.transpose("time", *spatial_dims).values.reshape(n_days, -1)
    O1, O2, MWmean = detect_rainy_season_onset(
        X,
        season_length=365,
        default_dry_threshold=1,
        wet_window_len=3,
        wet_threshold=20,
        dry_window_len=7,
        confirm_window_len=20,
        dry_threshold=0
    )
    O2_da = xr.DataArray(
        O2[0].reshape(spatial_shape),
        coords={d: da[d] for d in spatial_dims},
        dims=spatial_dims,
        name="rainy_season_onset"
    )
    sep16_doy = (pd.Timestamp(year, 9, 16) - pd.Timestamp(year, 1, 1)).days
    da_h2 = da.sel(time=da.time >= np.datetime64(f"{year}-09-16"))
    n_days_h2 = da_h2.sizes["time"]
    X_h2 = da_h2.transpose("time", *spatial_dims).values.reshape(n_days_h2, -1)
    O1_h2, O2_h2, MWmean_h2 = detect_rainy_season_onset(
        X_h2,
        season_length=n_days_h2,
        default_dry_threshold=1,
        wet_window_len=3,
        wet_threshold=20,
        dry_window_len=7,
        confirm_window_len=20,
        dry_threshold=0
    )
    O2_da_h2 = xr.DataArray(
        (O2_h2[0] + sep16_doy).reshape(spatial_shape),
        coords={d: da[d] for d in spatial_dims},
        dims=spatial_dims,
        name="rainy_season_onset_h2"
    )
    final = xr.merge([ds, O2_da.to_dataset(), O2_da_h2.to_dataset()])
    def day_to_date(x):
        if np.isnan(x):
            return pd.NaT
        else:
            return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(x))

    for var, name in [("rainy_season_onset",    "rainy_season_onset_date"),
                      ("rainy_season_onset_h2", "rainy_season_onset_date_h2")]:
        src = final[var]
        dates = xr.apply_ufunc(day_to_date, src, vectorize=True)
        dates = xr.DataArray(dates, coords=src.coords, dims=src.dims, name=name)
        final = xr.merge([final, dates])
    output_file = f"{output_dir}/ONSET_{year}.nc4"
    if os.path.isfile(output_file):
        os.remove(output_file)
    final.to_netcdf(output_file, mode="w", format="NETCDF4")
    print(f"Saved {output_file}")
    ds.close()

def process_sst_data(year):
    """
    Process ERA5 sea surface temperature (SST) data for a given year.

    This function:
    - Loads ERA5 SST data (in Kelvin) for a given year.
    - Renames coordinates to standard names (lat, lon).
    - Subsets to the western Indian Ocean region.
    - Aggregates hourly data to daily means.
    - Regrids data to a 1° x 1° grid.
    - Converts temperature from Kelvin to Celsius.
    - Saves output as a NetCDF file.

    Parameters:
    year (int): Year to process

    Input:
    ERA5 SST files (sstk variable) from: */ERA5/

    Output:
    $SCRATCH/sst/ERA5_{year}.nc4

    Variables:
    SSTK: sea surface temperature (Kelvin)
    SSTC: sea surface temperature (Celsius)

    Notes:
    - Assumes ERA5 naming convention (sstk variable).
    - Converts longitude/latitude naming for consistency.
    """
    output_dir = os.path.expandvars("$SCRATCH/sst")
    os.makedirs(output_dir, exist_ok=True)
    path = f"*/ERA5/*.nc"
    ds = xr.open_mfdataset(path, combine="by_coords")
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    lat_min, lat_max = -15, 5
    lon_min, lon_max = 45, 60
    ds_cropped = ds.sel(
        lat=slice(lat_max, lat_min),
        lon=slice(lon_min, lon_max)
    )
    ds_cropped_daily = ds_cropped.resample(time="1D").mean()
    target = xarray_regrid.Grid(
        north=lat_max,
        south=lat_min,
        east=lon_max,
        west=lon_min,
        resolution_lat=1,
        resolution_lon=1,
    ).create_regridding_dataset(lat_name="lat", lon_name="lon")
    ds_cropped_daily_regridded = ds_cropped_daily.regrid.stat(
        target,
        method="mean",
        time_dim="time",
        skipna=False
    )
    ds_cropped_daily_regridded["SSTC"] = ds_cropped_daily_regridded["SSTK"] - 273.15
    output_file = f"{output_dir}/ERA5_{year}.nc4"
    ds_cropped_daily_regridded.to_netcdf(output_file, mode="w", format="NETCDF4")
    print(f"Saved {output_file}")

def process_era5_data(year, variables):
    """
    Process ERA5 atmospheric variables for a given year.

    This function:
    - Loads ERA5 data for specified variables.
    - Merges them into a single dataset.
    - Renames coordinates to standard names (lat, lon).
    - Subsets to East Africa region.
    - Aggregates hourly data to daily means.
    - Regrids to a 1° x 1° grid.
    - Saves output as a NetCDF file.

    Parameters:
    year (int): Year to process
    variables (list of str): ERA5 variable names (e.g., ["2t", "sp", "tcwv"])

    Input:
    ERA5 files from project data directory

    Output:
    $SCRATCH/era5/ERA5_{year}.nc4

    Notes:
    - Variables must match ERA5 file naming conventions.
    - Assumes hourly ERA5 data.
    - Output is daily averaged and regridded.
    """
    output_dir = os.path.expandvars("$SCRATCH/era5")
    os.makedirs(output_dir, exist_ok=True)
    path_patterns = [f"*/ERA5/*/{year}*/*{var}*.nc" for var in variables]
    ds_list = [xr.open_mfdataset(p, combine="by_coords") for p in path_patterns]
    ds = xr.merge(ds_list)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    lat_min, lat_max = -10, 15
    lon_min, lon_max = 30, 52
    ds_cropped = ds.sel(
        lat=slice(lat_max, lat_min),
        lon=slice(lon_min, lon_max)
    )
    ds_cropped_daily = ds_cropped.resample(time="1D").mean()
    target = xarray_regrid.Grid(
        north=lat_max,
        south=lat_min,
        east=lon_max,
        west=lon_min,
        resolution_lat=1,
        resolution_lon=1,
    ).create_regridding_dataset(lat_name="lat", lon_name="lon")
    ds_cropped_daily_regridded = ds_cropped_daily.regrid.stat(
        target,
        method="mean",
        time_dim="time",
        skipna=False
    )
    output_file = f"{output_dir}/ERA5_{year}.nc4"
    ds_cropped_daily_regridded.to_netcdf(output_file, mode="w", format="NETCDF4")
    print(f"Saved {output_file}")