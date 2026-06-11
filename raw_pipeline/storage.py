from __future__ import annotations

from pathlib import Path

import pandas as pd

from raw_pipeline.config import IngestionSettings


class RawCsvStorage:
    """Persist raw source data in the existing data/raw CSV layout."""

    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
        self.settings.raw_air_dir.mkdir(parents=True, exist_ok=True)
        self.settings.raw_weather_dir.mkdir(parents=True, exist_ok=True)

    def save_air_quality(self, df: pd.DataFrame, month_tag: str) -> Path:
        return self._save(df, self.settings.raw_air_dir / f"air_quality_{month_tag}.csv")

    def save_weather(self, df: pd.DataFrame, month_tag: str) -> Path:
        return self._save(df, self.settings.raw_weather_dir / f"weather_{month_tag}.csv")

    @staticmethod
    def _save(df: pd.DataFrame, path: Path) -> Path:
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["datetime"])
            df = pd.concat([existing, df], ignore_index=True)

        df = (
            df.drop_duplicates(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        df.to_csv(path, index=False)
        return path
