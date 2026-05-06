from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

# Hanoi city center coordinates
LAT = 21.0285
LON = 105.8542

START_DATE = "2013-01-01"
END_DATE = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
WEATHER_CHUNK_DAYS = 20

BASE_DIR = Path(".")
RAW_AIR_DIR = BASE_DIR / "data" / "raw" / "air_quality"
RAW_WEATHER_DIR = BASE_DIR / "data" / "raw" / "weather"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

for p in [RAW_AIR_DIR, RAW_WEATHER_DIR, PROCESSED_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def month_ranges(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cur = start.replace(day=1)

    while cur <= end:
        month_start = max(cur, start)
        month_end = min(cur + pd.offsets.MonthEnd(0), end)
        yield month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
        cur = cur + pd.offsets.MonthBegin(1)


def date_chunks(start_date: str, end_date: str, chunk_days: int):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cur = start
    step = pd.Timedelta(days=chunk_days - 1)

    while cur <= end:
        chunk_end = min(cur + step, end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)


def get_json(url: str, params: dict, timeout: int = 90, retries: int = 4) -> dict:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        resp = None
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            detail = resp.text if resp is not None else ""
            try:
                detail = resp.json().get("reason", detail) if resp is not None else detail
            except ValueError:
                pass

            status_code = resp.status_code if resp is not None else None
            last_error = requests.HTTPError(f"{exc}. API detail: {detail}")
            retriable = status_code is not None and status_code >= 500

            if not retriable or attempt == retries:
                raise last_error from exc
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                raise

        time.sleep(min(2 * attempt, 6))

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unexpected request flow in get_json")


def hourly_payload_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    # Parse as UTC, then store naive UTC timestamps to match repo format.
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return df.rename(columns={"time": "datetime"})


def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    base_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "us_aqi",
                "european_aqi",
            ]
        ),
        "timezone": "GMT",
    }

    try:
        payload = get_json(url, {**base_params, "domains": "cams_europe"})
    except requests.HTTPError as exc:
        # cams_europe can return no coverage outside Europe (e.g., Hanoi).
        if "No data is available for this location" not in str(exc):
            raise
        payload = get_json(url, base_params)

    return hourly_payload_to_df(payload)


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    parts: list[pd.DataFrame] = []

    for chunk_start, chunk_end in date_chunks(start_date, end_date, WEATHER_CHUNK_DAYS):
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "dew_point_2m",
                    "pressure_msl",
                    "surface_pressure",
                    "precipitation",
                    "rain",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                ]
            ),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
        payload = get_json(url, params)
        parts.append(hourly_payload_to_df(payload))

    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    parts = [pd.read_csv(path, parse_dates=["datetime"]) for path in files]
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["datetime"]

    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    out["season"] = np.select(
        [
            out["month"].isin([12, 1, 2]),
            out["month"].isin([3, 4, 5]),
            out["month"].isin([6, 7, 8]),
            out["month"].isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "autumn"],
        default="unknown",
    )

    out["is_dry_season"] = out["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    return out


def add_pm_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("datetime").reset_index(drop=True)

    out["pm25_lag1"] = out["pm25"].shift(1)
    out["pm25_lag6"] = out["pm25"].shift(6)
    out["pm25_lag24"] = out["pm25"].shift(24)
    out["pm25_lag168"] = out["pm25"].shift(168)

    out["pm25_rolling_3h"] = out["pm25"].rolling(3, min_periods=3).mean()
    out["pm25_rolling_6h"] = out["pm25"].rolling(6, min_periods=6).mean()
    out["pm25_rolling_24h"] = out["pm25"].rolling(24, min_periods=24).mean()
    out["pm25_rolling_7d"] = out["pm25"].rolling(168, min_periods=168).mean()
    out["pm25_rolling_24h_std"] = out["pm25"].rolling(24, min_periods=24).std()

    out["pm10_lag1"] = out["pm10"].shift(1)
    out["pm10_lag24"] = out["pm10"].shift(24)

    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    radians = np.deg2rad(out["wind_direction"])
    out["wind_u"] = -out["wind_speed"] * np.sin(radians)
    out["wind_v"] = -out["wind_speed"] * np.cos(radians)

    out["temp_humidity"] = out["temperature"] * out["humidity"]
    out["pressure_diff"] = out["pressure_msl"].diff()
    out["is_raining"] = (out["rain"].fillna(0) > 0).astype(int)

    return out


def aqi_to_class(aqi_series: pd.Series) -> pd.Series:
    bins = [-np.inf, 50, 100, 150, 200, 300, np.inf]
    labels = [
        "good",
        "moderate",
        "unhealthy_sensitive",
        "unhealthy",
        "very_unhealthy",
        "hazardous",
    ]
    return pd.cut(aqi_series, bins=bins, labels=labels, ordered=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for h in [1, 6, 24]:
        out[f"target_pm25_t+{h}"] = out["pm25"].shift(-h)
        out[f"target_us_aqi_t+{h}"] = out["us_aqi"].shift(-h)
        out[f"target_aqi_class_t+{h}"] = aqi_to_class(out["us_aqi"].shift(-h))

    return out


def build_dataset(aq_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = aq_df.merge(weather_df, on="datetime", how="inner")

    df = df.rename(
        columns={
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
        }
    )

    df = add_time_features(df)
    df = add_pm_features(df)
    df = add_derived_features(df)
    df = add_targets(df)

    ordered_cols = [
        "datetime",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "co",
        "so2",
        "us_aqi",
        "european_aqi",
        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
        "is_weekend",
        "season",
        "is_dry_season",
        "temperature",
        "humidity",
        "dew_point",
        "pressure_msl",
        "surface_pressure",
        "precipitation",
        "rain",
        "cloud_cover",
        "wind_speed",
        "wind_direction",
        "wind_gusts",
        "wind_u",
        "wind_v",
        "temp_humidity",
        "pressure_diff",
        "is_raining",
        "pm25_lag1",
        "pm25_lag6",
        "pm25_lag24",
        "pm25_lag168",
        "pm25_rolling_3h",
        "pm25_rolling_6h",
        "pm25_rolling_24h",
        "pm25_rolling_7d",
        "pm25_rolling_24h_std",
        "pm10_lag1",
        "pm10_lag24",
        "target_pm25_t+1",
        "target_pm25_t+6",
        "target_pm25_t+24",
        "target_us_aqi_t+1",
        "target_us_aqi_t+6",
        "target_us_aqi_t+24",
        "target_aqi_class_t+1",
        "target_aqi_class_t+6",
        "target_aqi_class_t+24",
    ]
    return df[ordered_cols].sort_values("datetime").reset_index(drop=True)


TRAIN_REQUIRED_COLUMNS = [
    "pm25_lag1",
    "pm25_lag6",
    "pm25_lag24",
    "pm25_lag168",
    "pm25_rolling_3h",
    "pm25_rolling_6h",
    "pm25_rolling_24h",
    "pm25_rolling_7d",
    "pm25_rolling_24h_std",
    "pressure_diff",
    "target_pm25_t+1",
    "target_pm25_t+6",
    "target_pm25_t+24",
    "target_us_aqi_t+1",
    "target_us_aqi_t+6",
    "target_us_aqi_t+24",
    "target_aqi_class_t+1",
    "target_aqi_class_t+6",
    "target_aqi_class_t+24",
]


def write_processed_outputs(aq_all: pd.DataFrame, wt_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_df = build_dataset(aq_all, wt_all)
    save_csv(full_df, PROCESSED_DIR / "hanoi_air_ml_ready_full.csv")

    train_df = full_df.dropna(subset=TRAIN_REQUIRED_COLUMNS).reset_index(drop=True)
    save_csv(train_df, PROCESSED_DIR / "hanoi_air_ml_ready_train.csv")

    return full_df, train_df


def build_processed_from_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    aq_all = load_raw_csvs(RAW_AIR_DIR)
    wt_all = load_raw_csvs(RAW_WEATHER_DIR)
    return write_processed_outputs(aq_all, wt_all)


def fetch_raw_and_build_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    air_parts = []
    weather_parts = []

    for start_date, end_date in month_ranges(START_DATE, END_DATE):
        print(f"Fetching {start_date} -> {end_date}")

        aq = fetch_air_quality(start_date, end_date)
        wt = fetch_weather(start_date, end_date)

        air_parts.append(aq)
        weather_parts.append(wt)

        month_tag = pd.Timestamp(start_date).strftime("%Y_%m")
        save_csv(aq, RAW_AIR_DIR / f"air_quality_{month_tag}.csv")
        save_csv(wt, RAW_WEATHER_DIR / f"weather_{month_tag}.csv")

    aq_all = pd.concat(air_parts, ignore_index=True).drop_duplicates(subset=["datetime"])
    wt_all = pd.concat(weather_parts, ignore_index=True).drop_duplicates(subset=["datetime"])

    return write_processed_outputs(aq_all, wt_all)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Hanoi ML-ready air quality datasets.")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Build processed datasets from existing data/raw CSV files without fetching APIs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.from_raw:
        full_df, train_df = build_processed_from_raw()
    else:
        full_df, train_df = fetch_raw_and_build_processed()

    print("\nDone.")
    print(f"Rows full : {len(full_df):,}")
    print(f"Rows train: {len(train_df):,}")
    print(full_df.head())


if __name__ == "__main__":
    main()
