# Data-Science
Hệ thống dự báo ô nhiễm không khí

## Data Flow

```text
Cron / Airflow / Prefect
  -> raw_pipeline.orchestrators.*
  -> RawIngestionPipeline
  -> OpenMeteoCrawler
  -> RawCsvStorage
  -> data/raw/air_quality + data/raw/weather
```

Raw pipeline chỉ chịu trách nhiệm lấy và lưu dữ liệu gốc. Feature engineering và tạo ML-ready dataset vẫn nằm ở `build_hanoi_dataset.py`.

## Chạy Raw Pipeline

Chạy theo khoảng ngày cụ thể:

```bash
python -m raw_pipeline.orchestrators.cron --start-date 2026-04-01 --end-date 2026-04-30
```

Chạy dạng job định kỳ, lấy lại vài ngày gần nhất để tránh miss dữ liệu:

```bash
python -m raw_pipeline.orchestrators.cron --lookback-days 2
```

Cron example:

```cron
*/5 * * * * cd /path/to/Data-Science && python -m raw_pipeline.orchestrators.cron --lookback-days 2
```

Airflow DAG nằm ở `raw_pipeline/orchestrators/airflow_dag.py`. Prefect flow nằm ở `raw_pipeline/orchestrators/prefect_flow.py`.
