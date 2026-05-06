from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from raw_pipeline.config import IngestionSettings
from raw_pipeline.crawlers import OpenMeteoCrawler
from raw_pipeline.dates import month_ranges
from raw_pipeline.storage import RawCsvStorage


@dataclass(frozen=True)
class RawIngestionResult:
    start_date: str
    end_date: str
    air_rows: int
    weather_rows: int
    files_written: tuple[str, ...]


class RawIngestionPipeline:
    """Orchestrates: scheduler trigger -> crawler -> raw storage."""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        crawler: OpenMeteoCrawler | None = None,
        storage: RawCsvStorage | None = None,
    ) -> None:
        self.settings = settings or IngestionSettings.from_env()
        self.crawler = crawler or OpenMeteoCrawler(self.settings)
        self.storage = storage or RawCsvStorage(self.settings)

    def run(self, start_date: str, end_date: str) -> RawIngestionResult:
        air_rows = 0
        weather_rows = 0
        files_written: list[str] = []

        for month_start, month_end in month_ranges(start_date, end_date):
            month_tag = pd.Timestamp(month_start).strftime("%Y_%m")
            print(f"Ingesting raw data {month_start} -> {month_end}")

            air_df = self.crawler.fetch_air_quality(month_start, month_end)
            weather_df = self.crawler.fetch_weather(month_start, month_end)

            air_path = self.storage.save_air_quality(air_df, month_tag)
            weather_path = self.storage.save_weather(weather_df, month_tag)

            air_rows += len(air_df)
            weather_rows += len(weather_df)
            files_written.extend([str(air_path), str(weather_path)])

        return RawIngestionResult(
            start_date=start_date,
            end_date=end_date,
            air_rows=air_rows,
            weather_rows=weather_rows,
            files_written=tuple(files_written),
        )

    def run_recent(self, lookback_days: int | None = None) -> RawIngestionResult:
        days = lookback_days if lookback_days is not None else self.settings.default_lookback_days
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days)
        return self.run(start.isoformat(), end.isoformat())
