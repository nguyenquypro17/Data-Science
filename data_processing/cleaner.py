import pandas as pd

def clean_and_merge(aq_df, weather_df):
    aq_df = aq_df.drop_duplicates("datetime").sort_values("datetime")
    weather_df = weather_df.drop_duplicates("datetime").sort_values("datetime")

    df = pd.merge_asof(
        aq_df,
        weather_df,
        on="datetime",
        direction="backward",
        tolerance=pd.Timedelta("30min")
    )

    df = df.rename(columns={
        "pm2_5": "pm25", 
        "carbon_monoxide": "co", 
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2", 
        "ozone": "o3", 
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity", 
        "dew_point_2m": "dew_point",
        "wind_speed_10m": "wind_speed", 
        "wind_direction_10m": "wind_direction",
        "wind_gusts_10m": "wind_gusts",
    })

    df = df.sort_values("datetime")
    df = df.set_index("datetime").resample("1h").asfreq().reset_index()

    cols_cannot_be_negative = ["pm25", "pm10", "precipitation", "rain", "wind_speed", "co", "no2", "so2", "o3"]
    for col in cols_cannot_be_negative:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(0, 100)

    # Xử lý Missing Values
    numeric_cols = df.select_dtypes(include=["number"]).columns

    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit=3, limit_direction="forward"
    )

    df[numeric_cols] = df[numeric_cols].ffill(limit=24)
    df[numeric_cols] = df[numeric_cols].bfill(limit=3)

    assert df["datetime"].is_monotonic_increasing

    return df