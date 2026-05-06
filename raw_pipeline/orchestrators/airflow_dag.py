from __future__ import annotations

from datetime import datetime

from raw_pipeline.pipeline import RawIngestionPipeline

try:
    from airflow.decorators import dag, task
except ImportError as exc:  # pragma: no cover - Airflow is an optional runtime dependency.
    raise RuntimeError("Install Apache Airflow to load this DAG.") from exc


@dag(
    dag_id="hanoi_raw_ingestion",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hanoi", "air-quality", "raw-ingestion"],
)
def hanoi_raw_ingestion_dag():
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

    ingest_recent()


hanoi_raw_ingestion = hanoi_raw_ingestion_dag()
