from __future__ import annotations

import argparse

from raw_pipeline.config import IngestionSettings
from raw_pipeline.pipeline import RawIngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hanoi raw data pipeline.")
    parser.add_argument("--start-date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Run the recent raw pipeline for the last N days when start/end are not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = RawIngestionPipeline(IngestionSettings.from_env())

    if args.start_date and args.end_date:
        result = pipeline.run(args.start_date, args.end_date)
    elif args.start_date or args.end_date:
        raise SystemExit("--start-date and --end-date must be provided together")
    else:
        result = pipeline.run_recent(args.lookback_days)

    print(
        "Raw pipeline complete: "
        f"air_rows={result.air_rows:,}, "
        f"weather_rows={result.weather_rows:,}, "
        f"files={len(result.files_written)}"
    )


if __name__ == "__main__":
    main()
