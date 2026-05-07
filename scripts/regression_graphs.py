import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import pandas as pd

def rmse_map(model, df):
    """
    Generate a spatial RMSE map for OND onset predictions.

    This function:
    - Combines training and testing predictions.
    - Computes RMSE independently at each grid cell.
    - Constructs a latitude-longitude RMSE grid.
    - Produces a spatial visualization using Cartopy.

    Parameters:
    model (dict):
        Output dictionary returned by run_lasso_model().

    df (pd.DataFrame):
        Original regression DataFrame containing spatial coordinates.

    Returns:
    None

    Outputs:
    - Displays a spatial RMSE map.
    - Saves figure as:
        "model_rmse_spatial_map.png"

    Notes:
    - RMSE is computed separately for each latitude-longitude grid cell.
    - Uses PlateCarree projection.
    - Missing grid cells remain NaN.
    - Annotates each valid cell with RMSE values.
    """
    lat_full = np.arange(-1, 8)
    lon_full = np.arange(36, 50)
    proj = ccrs.PlateCarree()
    model_train_df = model["train_df"].copy()
    model_train_df["predictions"] = model["train_predictions"]
    model_test_df = model["test_df"].copy()
    model_test_df["predictions"] = model["test_predictions"]
    model_df = pd.concat([model_train_df, model_test_df])
    spatial_cols = ["lat", "lon", "ond_onset"]
    model_df = model_df.join(
        df[spatial_cols].loc[model_df.index], how="left", rsuffix="_orig"
    )
    per_cell_rmse = (
        model_df.groupby(["lat", "lon"])
        .apply(lambda g: np.sqrt(np.mean((g["ond_onset"] - g["predictions"]) ** 2)))
        .reset_index()
        .rename(columns={0: "rmse"})
    )
    pivot = (
        per_cell_rmse.pivot(index="lat", columns="lon", values="rmse")
        .reindex(index=lat_full, columns=lon_full)
    )
    R    = pivot.values
    lats = pivot.index.values
    lons = pivot.columns.values
    lon2d, lat2d = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(10, 6))
    ax  = plt.axes(projection=proj)
    pcm = ax.pcolormesh(
        lon2d, lat2d, R,
        cmap="YlOrRd",
        shading="auto", transform=proj
    )
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            val = R[i, j]
            if np.isnan(val):
                continue
            ax.text(
                lon, lat, f"{val:.1f}",
                ha="center", va="center",
                fontsize=6, color="black",
                transform=proj,
            )
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND,    facecolor="lightgrey", zorder=0)
    ax.add_feature(cfeature.OCEAN,   facecolor="lightblue", zorder=0)
    ax.set_extent(
        [lon_full[0]-0.5, lon_full[-1]+0.5,
         lat_full[0]-0.5, lat_full[-1]+0.5],
        crs=proj
    )
    ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="grey")
    plt.colorbar(pcm, ax=ax, label="RMSE (days)", shrink=0.85)
    ax.set_title("Per-Cell RMSE OND Onset Predictions", fontsize=12)
    plt.tight_layout()
    plt.savefig("model_rmse_spatial_map.png", dpi=300, bbox_inches="tight")
    plt.show()