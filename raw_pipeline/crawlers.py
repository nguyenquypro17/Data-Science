from __future__ import annotations

import pandas as pd
import requests

from raw_pipeline.config import IngestionSettings
from raw_pipeline.dates import date_chunks
from raw_pipeline.http_client import RetryingHttpClient


class OpenMeteoCrawler:
    """Crawler for the raw air quality and weather APIs used by this project."""

    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, settings: IngestionSettings, http_client: RetryingHttpClient | None = None) -> None:
        self.settings = settings
        self.http_client = http_client or RetryingHttpClient(
            timeout_seconds=settings.request_timeout_seconds,
            retries=settings.request_retries,
        )

    def fetch_air_quality(self, start_date: str, end_date: str) -> pd.DataFrame:
        base_params = {
            "latitude": self.settings.latitude,
            "longitude": self.settings.longitude,
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
            payload = self.http_client.get_json(self.AIR_QUALITY_URL, {**base_params, "domains": "cams_europe"})
        except requests.HTTPError as exc:
            if "No data is available for this location" not in str(exc):
                raise
            payload = self.http_client.get_json(self.AIR_QUALITY_URL, base_params)

        return _hourly_payload_to_df(payload)

    def fetch_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []

        for chunk_start, chunk_end in date_chunks(
            start_date,
            end_date,
            self.settings.weather_chunk_days,
        ):
            params = {
                "latitude": self.settings.latitude,
                "longitude": self.settings.longitude,
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
            payload = self.http_client.get_json(self.WEATHER_URL, params)
            parts.append(_hourly_payload_to_df(payload))

        return (
            pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )


def _hourly_payload_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return df.rename(columns={"time": "datetime"})
