from data_loader import load_land, load_ocean, load_onset, load_sm, load_sst
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def make_classification_df():
    """
    Construct a classification-ready DataFrame for OND onset timing prediction.

    This function:
    - Loads precipitation, soil moisture, SST, ocean, and land ERA5 datasets.
    - Subsets datasets to East Africa and nearby western Indian Ocean regions.
    - Computes dekadal (10-day) averages for August and September predictors.
    - Extracts local and large-scale environmental variables.
    - Builds a tabular dataset for machine learning classification tasks.

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

    Returns:
    pd.DataFrame:
        DataFrame containing:
        - Spatial coordinates
        - Year
        - OND onset target variable
        - Environmental predictor variables

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
    return df

def modify_classification_df(df):
    """
    Construct a classification-ready DataFrame for OND onset timing prediction.

    This function:
    - Loads precipitation, soil moisture, SST, ocean, and land ERA5 datasets.
    - Subsets datasets to East Africa and nearby western Indian Ocean regions.
    - Computes dekadal (10-day) averages for August and September predictors.
    - Extracts local and large-scale environmental variables.
    - Builds a tabular dataset for machine learning classification tasks.

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

    Returns:
    pd.DataFrame:
        DataFrame containing:
        - Spatial coordinates
        - Year
        - OND onset target variable
        - Environmental predictor variables

    Notes:
    - Iterates over all grid cells and years individually.
    - Uses spatial averaging for SST/ocean variables.
    - Uses local grid-cell values for land and precipitation variables.
    - Only finite onset values are retained.
    """
    train_df = df[df["year"] < 2021].copy()
    test_df = df[df["year"] >= 2021].copy()
    early_threshold = train_df["ond_onset"].quantile(0.33)
    print(early_threshold)
    late_threshold  = train_df["ond_onset"].quantile(0.67)
    print(late_threshold)
    train_df["category"] = pd.cut(
        train_df["ond_onset"],
        bins=[-np.inf, early_threshold, late_threshold, np.inf],
        labels=["Early", "Normal", "Late"]
    )
    test_df["category"] = pd.cut(
        test_df["ond_onset"],
        bins=[-np.inf, early_threshold, late_threshold, np.inf],
        labels=["Early", "Normal", "Late"]
    )
    train_df["early_binary"] = (train_df["category"] == "Early").astype(int)
    test_df["early_binary"] = (test_df["category"] == "Early").astype(int)
    train_df["normal_binary"] = (train_df["category"] == "Normal").astype(int)
    test_df["normal_binary"] = (test_df["category"] == "Normal").astype(int)
    train_df["late_binary"] = (train_df["category"] == "Late").astype(int)
    test_df["late_binary"] = (test_df["category"] == "Late").astype(int)
    return train_df, test_df

def run_binary_logistic_model(train_df, test_df, feature_cols, target_col="early_binary"):
    """
    Convert continuous OND onset dates into categorical onset classes.

    This function:
    - Splits the dataset into training and testing periods.
    - Computes tercile thresholds from training onset dates.
    - Categorizes onset timing into:
        - Early
        - Normal
        - Late
    - Generates binary classification targets for each category.

    Parameters:
    df (pd.DataFrame):
        Input classification dataset.

    Returns:
    tuple:
        (train_df, test_df)

        Each DataFrame contains:
        - onset category labels
        - binary target columns for one-vs-rest classification

    Notes:
    - Thresholds are computed only from training years.
    - Training years are before 2021.
    - Testing years are 2021 and later.
    """
    train_df_clean = train_df[feature_cols + [target_col]].dropna()
    test_df_clean = test_df[feature_cols + [target_col]].dropna()
    X_train = train_df_clean[feature_cols]
    y_train = train_df_clean[target_col]
    X_test = test_df_clean[feature_cols]
    y_test = test_df_clean[target_col]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegressionCV(
            cv=TimeSeriesSplit(),
            class_weight="balanced",
            solver="saga",
            max_iter=10000,
            penalty="l1"
        ))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    logreg = pipeline.named_steps["logreg"]
    scaler = pipeline.named_steps["scaler"]
    coef_df = pd.DataFrame(
        logreg.coef_,
        columns=feature_cols,
        index=["coef"]
    ).T.sort_values(by="coef", key=abs, ascending=False)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=logreg.classes_),
        "f1_score": f1_score(y_test, y_pred),
        "classes": logreg.classes_,
        "coef_df": coef_df,
        "test_df": test_df_clean,
        "train_df": train_df_clean,
        "test_predictions": y_pred,
        "train_predictions": pipeline.predict(X_train),
        "model": logreg,
        "scaler": scaler,
        "pipeline": pipeline
    }

def evaluate_logistic_model(results_dict, target_col="early_binary"):
    """
    Train and evaluate a binary logistic regression classifier.

    This function:
    - Standardizes predictor variables.
    - Fits an L1-regularized logistic regression model.
    - Uses time-series cross validation for hyperparameter selection.
    - Generates predictions on the testing dataset.
    - Computes classification metrics and feature coefficients.

    Parameters:
    train_df (pd.DataFrame):
        Training dataset.

    test_df (pd.DataFrame):
        Testing dataset.

    feature_cols (list of str):
        Predictor variable names.

    target_col (str):
        Binary target column name.
        Default is "early_binary".

    Returns:
    dict:
        Dictionary containing:
        - accuracy
        - classification report
        - confusion matrix
        - F1 score
        - feature coefficients
        - predictions
        - trained model objects

    Notes:
    - Uses L1 regularization for sparse feature selection.
    - Uses balanced class weights to mitigate class imbalance.
    - Uses saga solver for L1 optimization.
    """
    model = results_dict["model"]
    pipeline = results_dict["pipeline"]
    X_test = results_dict["test_df"].drop(columns=[target_col], errors="ignore")
    y_test = results_dict["test_df"][target_col]
    y_pred = results_dict["test_predictions"]
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("=== BASIC METRICS ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
    coefs = model.coef_.flatten()
    zero_coefs = np.sum(coefs == 0)
    print("\n=== LASSO SPARSITY ===")
    print(f"Total features: {len(coefs)}")
    print(f"Zero coefficients: {zero_coefs}")
    print(f"Sparsity: {zero_coefs / len(coefs):.2%}")
    coef_df = pd.DataFrame({
        "feature": results_dict["coef_df"].index,
        "coef": coefs
    }).sort_values(by="coef", key=abs, ascending=False)
    print("\n=== TOP FEATURES ===")
    print(coef_df)