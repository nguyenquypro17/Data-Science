from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionSettings:
    """Runtime settings for the raw data pipeline."""

    base_dir: Path = Path(".")
    latitude: float = 21.0285
    longitude: float = 105.8542
    request_timeout_seconds: int = 90
    request_retries: int = 4
    weather_chunk_days: int = 20
    default_lookback_days: int = 2

    @property
    def raw_air_dir(self) -> Path:
        return self.base_dir / "data" / "raw" / "air_quality"

    @property
    def raw_weather_dir(self) -> Path:
        return self.base_dir / "data" / "raw" / "weather"

    @classmethod
    def from_env(cls) -> "IngestionSettings":
        return cls(
            base_dir=Path(os.getenv("HANOI_AIR_BASE_DIR", ".")),
            latitude=float(os.getenv("HANOI_AIR_LAT", "21.0285")),
            longitude=float(os.getenv("HANOI_AIR_LON", "105.8542")),
            request_timeout_seconds=int(os.getenv("HANOI_AIR_TIMEOUT_SECONDS", "90")),
            request_retries=int(os.getenv("HANOI_AIR_REQUEST_RETRIES", "4")),
            weather_chunk_days=int(os.getenv("HANOI_AIR_WEATHER_CHUNK_DAYS", "20")),
            default_lookback_days=int(os.getenv("HANOI_AIR_LOOKBACK_DAYS", "2")),
        )
