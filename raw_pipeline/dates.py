from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def month_ranges(start_date: str, end_date: str) -> Iterator[tuple[str, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cur = start.replace(day=1)

    while cur <= end:
        month_start = max(cur, start)
        month_end = min(cur + pd.offsets.MonthEnd(0), end)
        yield month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
        cur = cur + pd.offsets.MonthBegin(1)


def date_chunks(start_date: str, end_date: str, chunk_days: int) -> Iterator[tuple[str, str]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cur = start
    step = pd.Timedelta(days=chunk_days - 1)

    while cur <= end:
        chunk_end = min(cur + step, end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)
