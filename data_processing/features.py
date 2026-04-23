import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["datetime"]
    out["year"], out["month"], out["day"], out["hour"] = dt.dt.year, dt.dt.month, dt.dt.day, dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    
    out["season"] = np.select(
        [out["month"].isin([12, 1, 2]), out["month"].isin([3, 4, 5]), out["month"].isin([6, 7, 8]), out["month"].isin([9, 10, 11])],
        ["winter", "spring", "summer", "autumn"], default="unknown"
    )
    out["is_dry_season"] = out["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    return out

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    
    out['month_sin'] = np.sin(2 * np.pi * (out['month'] - 1) / 12)
    out['month_cos'] = np.cos(2 * np.pi * (out['month'] - 1) / 12)
    
    out['day_of_week_sin'] = np.sin(2 * np.pi * out['day_of_week'] / 7)
    out['day_of_week_cos'] = np.cos(2 * np.pi * out['day_of_week'] / 7)
    
    return out

def add_pm_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("datetime").reset_index(drop=True)
    
    for lag in [1, 6, 24, 168]:
        out[f"pm25_lag{lag}"] = out["pm25"].shift(lag)
    
    for window in [3, 6, 24]:
        out[f"pm25_rolling_{window}h"] = out["pm25"].rolling(
            window, min_periods=window, closed="left"
        ).mean()
        
    out["pm25_rolling_7d"] = out["pm25"].rolling(
        168, min_periods=168, closed="left"
    ).mean()
    
    out["pm25_rolling_24h_std"] = out["pm25"].rolling(
        24, min_periods=24, closed="left"
    ).std()
    
    return out

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    radians = np.deg2rad(out["wind_direction"])
    out["wind_u"] = -out["wind_speed"] * np.sin(radians)
    out["wind_v"] = -out["wind_speed"] * np.cos(radians)
    out["temp_humidity"] = out["temperature"] * out["humidity"]
    out["pressure_diff"] = out["pressure_msl"].diff().fillna(0)
    out["is_raining"] = (out["rain"].fillna(0) > 0).astype(int)
    return out

def aqi_to_class(aqi_series: pd.Series) -> pd.Series:
    bins = [-np.inf, 50, 100, 150, 200, 300, np.inf]
    labels = ["good", "moderate", "unhealthy_sensitive", "unhealthy", "very_unhealthy", "hazardous"]
    return pd.cut(aqi_series, bins=bins, labels=labels, ordered=True)

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for h in [1, 6, 24]:
        out[f"target_pm25_t+{h}"] = out["pm25"].shift(-h)
        out[f"target_us_aqi_t+{h}"] = out["us_aqi"].shift(-h)
        out[f"target_aqi_class_t+{h}"] = aqi_to_class(out["us_aqi"].shift(-h))
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_pm_features(df)
    df = add_derived_features(df)
    df = add_targets(df)
    target_class_cols = [c for c in df.columns if "target_aqi_class" in c]
    for col in target_class_cols:
        # Chuyển sang category và lấy code (0, 1, 2...)
        df[col] = pd.Categorical(df[col], categories=[
            "good", "moderate", "unhealthy_sensitive", 
            "unhealthy", "very_unhealthy", "hazardous"
        ]).codes
    df = pd.get_dummies(df, columns=['season'], prefix='season', drop_first=False)
    return df