import glob
import os
import xarray as xr

def load_onset():
    """
    Load precipitation onset dataset from NetCDF files.

    This function:
    - Searches for all NetCDF (.nc4) files in the onset directory on SCRATCH.
    - Sorts files to ensure consistent temporal ordering.
    - Opens them as a single combined xarray Dataset using coordinate alignment.

    Returns:
    xr.Dataset:
        Combined dataset containing precipitation onset-related variables across all files.

    Notes:
    - Uses xarray.open_mfdataset with combine="by_coords" to align data by coordinates.
    - Assumes files share compatible grid structure and metadata.
    """
    files = sorted(glob.glob(os.path.expandvars("$SCRATCH/onset/*.nc4")))
    return xr.open_mfdataset(files, combine="by_coords")

def load_sm():
    """
    Load soil moisture dataset from NetCDF files.

    This function:
    - Searches for all NetCDF (.nc4) files in the soil moisture directory on SCRATCH.
    - Sorts files for consistent ordering across time or ensemble members.
    - Opens and merges them into a single xarray Dataset aligned by coordinates.

    Returns:
    xr.Dataset:
        Combined soil moisture dataset across all available files.

    Notes:
    - Assumes consistent spatial grid and variable naming across files.
    - Coordinate-based merging ensures proper alignment of time/space dimensions.
    """
    files = sorted(glob.glob(os.path.expandvars("$SCRATCH/soil_moisture/*.nc4")))
    return xr.open_mfdataset(files, combine="by_coords")

def load_sst():
    """
    Load sea surface temperature (SST) dataset from NetCDF files.

    This function:
    - Searches for all NetCDF (.nc4) files in the SST directory on SCRATCH.
    - Sorts files to preserve consistent temporal ordering.
    - Opens and combines all files into a single xarray Dataset using coordinate alignment.

    Returns:
    xr.Dataset:
        Combined SST dataset spanning all available input files.

    Notes:
    - Uses xarray.open_mfdataset with combine="by_coords".
    - Requires consistent latitude/longitude grids across input files.
    """
    files = sorted(glob.glob(os.path.expandvars("$SCRATCH/sst/*.nc4")))
    return xr.open_mfdataset(files, combine="by_coords")

def load_ocean():
    """
    Load ERA5 ocean-atmosphere dataset from NetCDF files.

    This function:
    - Searches for all NetCDF (.nc4) files in the ERA5 ocean directory on SCRATCH.
    - Sorts files to preserve consistent temporal ordering.
    - Opens and combines all files into a single xarray Dataset using coordinate alignment.

    Returns:
    xr.Dataset:
        Combined ERA5 ocean dataset containing all available variables and times.

    Notes:
    - Uses xarray.open_mfdataset with combine="by_coords".
    - Assumes files share compatible coordinates, dimensions, and metadata.
    - Coordinate-based merging automatically aligns overlapping dimensions such as time, latitude, and longitude.
    """
    files = sorted(glob.glob(os.path.expandvars("$SCRATCH/era5_ocean/*.nc4")))
    return xr.open_mfdataset(files, combine="by_coords")

def load_land():
    """
    Load ERA5 land dataset from NetCDF files.

    This function:
    - Searches for all NetCDF (.nc4) files in the ERA5 land directory on SCRATCH.
    - Sorts files to ensure consistent ordering across time and variables.
    - Opens and merges all files into a single xarray Dataset using coordinate alignment.

    Returns:
    xr.Dataset:
        Combined ERA5 land dataset spanning all available files.

    Notes:
    - Uses xarray.open_mfdataset with combine="by_coords".
    - Assumes all files use compatible spatial grids and coordinate conventions.
    - Coordinate alignment ensures proper merging of shared dimensions.
    """
    files = sorted(glob.glob(os.path.expandvars("$SCRATCH/era5_land/*.nc4")))
    return xr.open_mfdataset(files, combine="by_coords")