from data_loader import load_onset, load_sm, load_sst, load_ocean, load_land
import numpy as np
from itertools import product
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

def make_regression_df():
    """
    Construct a regression-ready DataFrame for OND onset prediction.

    This function:
    - Loads precipitation, soil moisture, SST, ocean, and land ERA5 datasets.
    - Subsets datasets to East Africa and nearby western Indian Ocean regions.
    - Computes dekadal (10-day) averages for August and September predictors.
    - Extracts spatially varying and large-scale environmental predictors.
    - Builds a tabular dataset for machine learning regression.

    Predictors include:
    - SST
    - Soil moisture
    - Root-zone soil moisture
    - Precipitation
    - Total column water vapor
    - Mean sea level pressure
    - 10 m zonal and meridional wind
    - 2 m temperature and dewpoint
    - CAPE
    - Total cloud cover

    Additional engineered features include:
    - Square root transforms
    - Polynomial spatial terms

    Returns:
    pd.DataFrame:
        DataFrame containing:
        - Spatial coordinates
        - Year
        - OND onset target variable
        - Environmental predictor variables
        - Engineered nonlinear features

    Notes:
    - Iterates over all grid cells and years individually.
    - Uses spatial averaging for SST/ocean variables.
    - Uses local grid-cell values for land and precipitation variables.
    - Only finite onset values are retained.
    """
    precipitation = load_onset.sel(lat=slice(0, 6), lon=slice(37, 48))
    sm = load_sm.sel(lat=slice(0, 6), lon=slice(37, 48))
    sst = load_sst.sel(lat=slice(-15, 5), lon=slice(45, 60))
    ocean = load_ocean.sel(lat=slice(-15, 5), lon=slice(45, 60))
    land = load_land.sel(lat=slice(0, 6), lon=slice(37, 48))
    years = np.arange(2016, 2023)
    sm_lat = sm["lat"].values
    sm_lon = sm["lon"].values
    ond_onsets = []
    sep_precip_dekad1 = []
    sep_sst_dekad1 = []
    sep_sm_dekad1 = []
    sep_rzsm_dekad1 = []
    onset_years = []
    lats = []
    lons = []
    aug_precip_dekad1, aug_precip_dekad2, aug_precip_dekad3 = [], [], []
    aug_sst_dekad1,    aug_sst_dekad2,    aug_sst_dekad3    = [], [], []
    aug_sm_dekad1,     aug_sm_dekad2,     aug_sm_dekad3     = [], [], []
    aug_rzsm_dekad1,   aug_rzsm_dekad2,   aug_rzsm_dekad3   = [], [], []
    sep_tcwv_dekad1 = []
    sep_msl_dekad1 = []
    sep_10u_dekad1 = []
    sep_10v_dekad1 = []
    sep_2t_dekad1 = []
    sep_2d_dekad1 = []
    sep_cape_dekad1 = []
    sep_tcc_dekad1 = []
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
                aug_sst_dekad1.append(float(np.nanmean(sst["SSTC"].sel(time=slice(*dekads["aug_dekad1"])))))
                aug_sst_dekad2.append(float(np.nanmean(sst["SSTC"].sel(time=slice(*dekads["aug_dekad2"])))))
                aug_sst_dekad3.append(float(np.nanmean(sst["SSTC"].sel(time=slice(*dekads["aug_dekad3"])))))
                aug_sm_dekad1.append(float(np.nanmean(sm["soil_moisture"].sel(time=slice(*dekads["aug_dekad1"]), lat=lat, lon=lon))))
                aug_sm_dekad2.append(float(np.nanmean(sm["soil_moisture"].sel(time=slice(*dekads["aug_dekad2"]), lat=lat, lon=lon))))
                aug_sm_dekad3.append(float(np.nanmean(sm["soil_moisture"].sel(time=slice(*dekads["aug_dekad3"]), lat=lat, lon=lon))))
                aug_rzsm_dekad1.append(float(np.nanmean(sm["root_zone_soil_moisture"].sel(time=slice(*dekads["aug_dekad1"]), lat=lat, lon=lon))))
                aug_rzsm_dekad2.append(float(np.nanmean(sm["root_zone_soil_moisture"].sel(time=slice(*dekads["aug_dekad2"]), lat=lat, lon=lon))))
                aug_rzsm_dekad3.append(float(np.nanmean(sm["root_zone_soil_moisture"].sel(time=slice(*dekads["aug_dekad3"]), lat=lat, lon=lon))))
                aug_precip_dekad1.append(float(np.nanmean(precipitation["precipitation"].sel(time=slice(*dekads["aug_dekad1"]), lat=lat, lon=lon))))
                aug_precip_dekad2.append(float(np.nanmean(precipitation["precipitation"].sel(time=slice(*dekads["aug_dekad2"]), lat=lat, lon=lon))))
                aug_precip_dekad3.append(float(np.nanmean(precipitation["precipitation"].sel(time=slice(*dekads["aug_dekad3"]), lat=lat, lon=lon))))
                sep_sst_dekad1.append(float(np.nanmean(sst["SSTC"].sel(time=slice(*dekads["sep_dekad1"])))))
                sep_sm_dekad1.append(float(np.nanmean(sm["soil_moisture"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                sep_rzsm_dekad1.append(float(np.nanmean(sm["root_zone_soil_moisture"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                sep_precip_dekad1.append(float(np.nanmean(precipitation["precipitation"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                sep_tcwv_dekad1.append(float(np.nanmean(ocean["TCWV"].sel(time=slice(*dekads["sep_dekad1"])))))
                sep_msl_dekad1.append(float(np.nanmean(ocean["MSL"].sel(time=slice(*dekads["sep_dekad1"])))))
                sep_10u_dekad1.append(float(np.nanmean(ocean["VAR_10U"].sel(time=slice(*dekads["sep_dekad1"])))))
                sep_10v_dekad1.append(float(np.nanmean(ocean["VAR_10V"].sel(time=slice(*dekads["sep_dekad1"])))))
                sep_2t_dekad1.append(float(np.nanmean(land["VAR_2T"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                sep_2d_dekad1.append(float(np.nanmean(land["VAR_2D"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                sep_cape_dekad1.append(float(np.nanmean(land["CAPE"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                sep_tcc_dekad1.append(float(np.nanmean(land["TCC"].sel(time=slice(*dekads["sep_dekad1"]), lat=lat, lon=lon))))
                ond_onsets.append(onset)
                onset_years.append(year)
                lats.append(lat)
                lons.append(lon)
    df = pd.DataFrame({
        "year": onset_years, "lat": lats, "lon": lons,
        "ond_onset": ond_onsets,
        "aug_sst_dekad1": aug_sst_dekad1, "aug_sst_dekad2": aug_sst_dekad2, "aug_sst_dekad3": aug_sst_dekad3,
        "aug_sm_dekad1":  aug_sm_dekad1,  "aug_sm_dekad2":  aug_sm_dekad2,  "aug_sm_dekad3":  aug_sm_dekad3,
        "aug_rzsm_dekad1":aug_rzsm_dekad1,"aug_rzsm_dekad2":aug_rzsm_dekad2,"aug_rzsm_dekad3":aug_rzsm_dekad3,
        "aug_precip_dekad1":aug_precip_dekad1,"aug_precip_dekad2":aug_precip_dekad2,"aug_precip_dekad3":aug_precip_dekad3,
        "sep_sst_dekad1": sep_sst_dekad1,
        "sep_sm_dekad1":  sep_sm_dekad1,
        "sep_rzsm_dekad1":sep_rzsm_dekad1,
        "sep_precip_dekad1":sep_precip_dekad1,
        "sep_tcwv_dekad1": sep_tcwv_dekad1,
        "sep_msl_dekad1": sep_msl_dekad1,
        "sep_10u_dekad1": sep_10u_dekad1,
        "sep_10v_dekad1": sep_10v_dekad1,
        "sep_2t_dekad1": sep_2t_dekad1,
        "sep_2d_dekad1": sep_2d_dekad1,
        "sep_cape_dekad1": sep_cape_dekad1,
        "sep_tcc_dekad1": sep_tcc_dekad1
    })
    df["sep_precip_dekad1_square_root"] = df["sep_precip_dekad1"] ** (1/2)
    df["sep_sst_dekad1_square_root"] = df["sep_sst_dekad1"] ** (1/2)
    df["sep_rzsm_dekad1_squared"] = df["sep_rzsm_dekad1"] ** (2)
    df["lat_squared"] = df["lat"] ** (2)
    df["lon_seventh_power"] = df["lon"] ** (7)
    df = df[df["year"] >= 2016]
    return df

def run_lasso_model(df, feature_cols, target_col="ond_onset"):
    """
    Train and evaluate a LASSO regression model for OND onset prediction.

    This function:
    - Splits data into training and testing periods using year thresholds.
    - Standardizes predictor variables.
    - Performs LASSO regression with cross-validated regularization strength.
    - Evaluates model performance using RMSE and R² metrics.
    - Returns fitted model objects, predictions, and feature coefficients.

    Parameters:
    df (pd.DataFrame):
        Input regression dataset.

    feature_cols (list of str):
        Predictor variable column names.

    target_col (str):
        Target variable column name.
        Default is "ond_onset".

    Returns:
    dict:
        Dictionary containing:
        - rmse (float): Root mean squared prediction error
        - r2 (float): Coefficient of determination
        - coef_df (pd.DataFrame): Ranked LASSO coefficients
        - test_df (pd.DataFrame): Test dataset
        - train_df (pd.DataFrame): Training dataset
        - test_predictions (np.ndarray): Predicted test values
        - train_predictions (np.ndarray): Predicted training values
        - model (LassoCV): Trained LASSO estimator
        - scaler (StandardScaler): Fitted feature scaler
        - pipeline (Pipeline): Full sklearn pipeline

    Notes:
    - Uses time-aware cross validation via TimeSeriesSplit.
    - Training years are before 2021.
    - Testing years are 2021–2022.
    - Standardization is applied before regression.
    """
    train_df = df[df['year'] < 2021][feature_cols + [target_col]].dropna().copy()
    test_df = df[(df['year'] == 2022) | (df['year'] == 2021)][feature_cols + [target_col]].dropna().copy()
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=42, max_iter=20000))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    lasso = pipeline.named_steps["lasso"]
    scaler = pipeline.named_steps["scaler"]
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": lasso.coef_
    }).sort_values("coefficient", key=abs, ascending=False)
    return {
        "rmse": rmse,
        "r2": r2,
        "coef_df": coef_df,
        "test_df": test_df,
        "train_df": train_df,
        "test_predictions": y_pred,
        "train_predictions": pipeline.predict(X_train), 
        "model": lasso,
        "scaler": scaler,
        "pipeline": pipeline
    }