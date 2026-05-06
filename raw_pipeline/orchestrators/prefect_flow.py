from __future__ import annotations

from raw_pipeline.pipeline import RawIngestionPipeline

try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover - Prefect is an optional runtime dependency.
    raise RuntimeError("Install Prefect to run this flow.") from exc


@task
def ingest_recent() -> dict[str, int | str]:
    result = RawIngestionPipeline().run_recent()
    return {
        "start_date": result.start_date,
        "end_date": result.end_date,
        "air_rows": result.air_rows,
        "weather_rows": result.weather_rows,
        "files_written": len(result.files_written),
    }


@flow(name="hanoi-raw-ingestion")
def hanoi_raw_ingestion_flow() -> dict[str, int | str]:
    return ingest_recent()


if __name__ == "__main__":
    hanoi_raw_ingestion_flow()
