from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_gijon_aqi_hourly import (  # noqa: E402
    BASE_SIGNAL_COLS,
    TARGET_COL,
    WEATHER_COLS,
    add_lag_and_rolling_features,
    add_time_features,
    aqi_category,
)

APP_TITLE = "AirSense Hanoi AQI Forecast API"
LATITUDE = 21.0285
LONGITUDE = 105.8542
TIMEZONE = "Asia/Bangkok"
DEFAULT_HISTORY_DAYS = 10
MIN_HISTORY_HOURS = 170

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gijon_aqi_hourly"
BEST_MODELS_PATH = OUTPUT_DIR / "reports" / "best_models.json"
METRICS_PATH = OUTPUT_DIR / "reports" / "metrics_by_horizon.csv"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_HOURLY_VARS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "european_aqi",
]

WEATHER_HOURLY_VARS = [
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

AIR_RENAMES = {
    "pm2_5": "pm25",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
}

WEATHER_RENAMES = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "dew_point_2m": "dew_point",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gusts",
}


@dataclass(frozen=True)
class LoadedModel:
    horizon: int
    model_name: str
    model_path: Path
    metadata_path: Path
    rmse: float
    mae: float
    r2: float
    model: Any
    feature_cols: list[str]


app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_CACHE: dict[int, LoadedModel] | None = None


def resolve_project_path(value: str | Path) -> Path:
    raw = str(value).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_file(path: Path) -> Any:
    if path.suffix == ".joblib":
        return joblib.load(path)
    with path.open("rb") as f:
        return pickle.load(f)


def fitted_feature_names(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(name) for name in step.feature_names_in_]
    return []


def load_best_models() -> dict[int, LoadedModel]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    best_models = load_json(BEST_MODELS_PATH)
    loaded: dict[int, LoadedModel] = {}
    for horizon_text, record in best_models.items():
        horizon = int(horizon_text)
        model_path = resolve_project_path(record["model_path"])
        metadata_path = resolve_project_path(record["metadata_path"])
        metadata = load_json(metadata_path)
        model = load_model_file(model_path)
        feature_cols = fitted_feature_names(model) or metadata.get("feature_cols", [])
        if not feature_cols:
            raise RuntimeError(
                f"Best model for horizon {horizon} does not expose feature_cols. "
                "This app currently serves supervised MLP/SVMR/MARS models."
            )
        loaded[horizon] = LoadedModel(
            horizon=horizon,
            model_name=str(record["model_name"]),
            model_path=model_path,
            metadata_path=metadata_path,
            rmse=float(record["rmse"]),
            mae=float(record["mae"]),
            r2=float(record["r2"]),
            model=model,
            feature_cols=list(feature_cols),
        )

    _MODEL_CACHE = loaded
    return loaded


def get_json(url: str, params: dict[str, Any], timeout: int = 60, retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        response = None
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(attempt * 2, 6))

    raise RuntimeError(f"Open-Meteo request failed: {last_error}")


def hourly_payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo response does not contain hourly data.")
    frame = pd.DataFrame(hourly)
    frame["datetime"] = pd.to_datetime(frame["time"], utc=True)
    return frame.drop(columns=["time"])


def fetch_air_quality(history_days: int) -> pd.DataFrame:
    payload = get_json(
        AIR_QUALITY_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(AIR_HOURLY_VARS),
            "past_days": history_days,
            "forecast_days": 1,
            "timezone": "GMT",
        },
    )
    frame = hourly_payload_to_frame(payload).rename(columns=AIR_RENAMES)
    return frame


def fetch_weather(history_days: int) -> pd.DataFrame:
    payload = get_json(
        WEATHER_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(WEATHER_HOURLY_VARS),
            "past_days": history_days,
            "forecast_days": 1,
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        },
    )
    frame = hourly_payload_to_frame(payload).rename(columns=WEATHER_RENAMES)
    return frame


def add_runtime_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    numeric_cols = [
        col
        for col in [TARGET_COL, *BASE_SIGNAL_COLS, *WEATHER_COLS, "european_aqi"]
        if col in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    radians = np.deg2rad(out["wind_direction"])
    out["wind_u"] = -out["wind_speed"] * np.sin(radians)
    out["wind_v"] = -out["wind_speed"] * np.cos(radians)
    out["temp_humidity"] = out["temperature"] * out["humidity"]
    out["pressure_diff"] = out["pressure_msl"].diff()
    out["is_raining"] = (out["rain"].fillna(0) > 0).astype(int)

    fill_cols = [col for col in [*BASE_SIGNAL_COLS, *WEATHER_COLS, "european_aqi"] if col in out.columns]
    out[fill_cols] = out[fill_cols].interpolate(limit=3, limit_direction="both").ffill().bfill()
    return out


def fetch_live_hanoi_frame(history_days: int) -> pd.DataFrame:
    air = fetch_air_quality(history_days)
    weather = fetch_weather(history_days)
    merged = air.merge(weather, on="datetime", how="inner")

    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    merged = merged[merged["datetime"] <= now_utc].copy()
    if merged.empty:
        raise RuntimeError("No current or historical rows returned by Open-Meteo.")

    required = [TARGET_COL, "pm25", "pm10", "no2", "o3", "co", "so2"]
    missing = [col for col in required if col not in merged.columns]
    if missing:
        raise RuntimeError(f"Open-Meteo response is missing required columns: {missing}")

    frame = add_runtime_derived_features(merged)
    frame["datetime_utc"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["datetime_local"] = frame["datetime_utc"].dt.tz_convert(TIMEZONE)
    return frame.sort_values("datetime_utc").reset_index(drop=True)


def build_inference_feature_row(frame: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    enriched = add_time_features(frame)
    enriched = add_lag_and_rolling_features(enriched)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)

    required_features = list(dict.fromkeys(feature_cols))
    missing = [col for col in required_features if col not in enriched.columns]
    if missing:
        raise RuntimeError(f"Feature engineering did not create required columns: {missing[:10]}")

    ready = enriched.dropna(subset=required_features).reset_index(drop=True)
    if ready.empty:
        raise RuntimeError(
            f"Not enough complete hourly history. Need at least {MIN_HISTORY_HOURS} hours "
            "after API merge and feature engineering."
        )
    return ready.iloc[-1]


def prediction_frame_from_row(row: pd.Series, feature_cols: list[str]) -> np.ndarray:
    values: list[float] = []
    for col in feature_cols:
        value = row[col]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        values.append(float(value))
    return np.array([values])


def category_payload(aqi: float) -> dict[str, str]:
    category = str(aqi_category(np.array([aqi]))[0])
    mapping = {
        "good": {
            "label": "Tốt",
            "color": "#2fbf71",
            "severity": "low",
            "summary": "Không khí tốt, phù hợp cho hầu hết hoạt động ngoài trời.",
            "advice": "Có thể sinh hoạt ngoài trời bình thường.",
        },
        "moderate": {
            "label": "Trung bình",
            "color": "#f2c94c",
            "severity": "watch",
            "summary": "Không khí chấp nhận được, nhóm nhạy cảm nên theo dõi thêm.",
            "advice": "Người nhạy cảm nên giảm hoạt động ngoài trời kéo dài.",
        },
        "unhealthy_sensitive": {
            "label": "Kém cho nhóm nhạy cảm",
            "color": "#f2994a",
            "severity": "caution",
            "summary": "Nhóm nhạy cảm có thể bị ảnh hưởng sức khỏe.",
            "advice": "Trẻ em, người già, người có bệnh hô hấp nên hạn chế vận động ngoài trời.",
        },
        "unhealthy": {
            "label": "Xấu",
            "color": "#eb5757",
            "severity": "high",
            "summary": "Mọi người có thể bắt đầu bị ảnh hưởng sức khỏe.",
            "advice": "Giảm thời gian ngoài trời; cân nhắc khẩu trang lọc bụi mịn khi cần di chuyển.",
        },
        "very_unhealthy": {
            "label": "Rất xấu",
            "color": "#9b51e0",
            "severity": "very_high",
            "summary": "Nguy cơ sức khỏe cao cho toàn bộ dân cư.",
            "advice": "Tránh hoạt động ngoài trời; đóng cửa sổ và dùng máy lọc không khí nếu có.",
        },
        "hazardous": {
            "label": "Nguy hại",
            "color": "#7b1f4d",
            "severity": "hazardous",
            "summary": "Cảnh báo sức khỏe khẩn cấp.",
            "advice": "Ở trong nhà, tránh vận động ngoài trời và theo dõi khuyến cáo y tế.",
        },
    }
    return {"key": category, **mapping[category]}


def forecast_from_live_frame(frame: pd.DataFrame) -> dict[str, Any]:
    models = load_best_models()
    forecasts: list[dict[str, Any]] = []

    for horizon in sorted(models):
        loaded = models[horizon]
        row = build_inference_feature_row(frame, loaded.feature_cols)
        x = prediction_frame_from_row(row, loaded.feature_cols)
        pred = float(np.asarray(loaded.model.predict(x)).reshape(-1)[0])
        pred = float(np.clip(pred, 0, 500))
        target_time = row["datetime_local"] + pd.Timedelta(hours=horizon)
        category = category_payload(pred)

        forecasts.append(
            {
                "horizon_hours": horizon,
                "model": loaded.model_name,
                "aqi": round(pred, 1),
                "aqi_low": round(max(0.0, pred - loaded.rmse), 1),
                "aqi_high": round(min(500.0, pred + loaded.rmse), 1),
                "target_time_local": target_time.isoformat(),
                "origin_time_local": row["datetime_local"].isoformat(),
                "category": category,
                "metrics": {
                    "rmse": round(loaded.rmse, 3),
                    "mae": round(loaded.mae, 3),
                    "r2": round(loaded.r2, 4),
                },
            }
        )

    latest = frame.iloc[-1]
    current_aqi = float(latest[TARGET_COL])
    recent = frame.tail(72)
    return {
        "location": {
            "name": "Hà Nội",
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": TIMEZONE,
        },
        "data_source": {
            "air_quality": "Open-Meteo Air Quality API",
            "weather": "Open-Meteo Forecast API",
            "history_hours": int(len(frame)),
            "note": "Model được train cho Hà Nội; không dùng traffic feature.",
        },
        "current": {
            "time_local": latest["datetime_local"].isoformat(),
            "us_aqi": round(current_aqi, 1),
            "category": category_payload(current_aqi),
            "pm25": round(float(latest["pm25"]), 2),
            "pm10": round(float(latest["pm10"]), 2),
            "no2": round(float(latest["no2"]), 2),
            "o3": round(float(latest["o3"]), 2),
            "temperature": round(float(latest["temperature"]), 1),
            "humidity": round(float(latest["humidity"]), 1),
            "wind_speed": round(float(latest["wind_speed"]), 2),
            "rain": round(float(latest["rain"]), 2),
        },
        "forecasts": forecasts,
        "recent_observations": [
            {
                "time_local": row.datetime_local.isoformat(),
                "us_aqi": round(float(row.us_aqi), 1),
                "pm25": round(float(row.pm25), 2),
                "pm10": round(float(row.pm10), 2),
            }
            for row in recent.itertuples(index=False)
        ],
    }


def read_metrics() -> list[dict[str, Any]]:
    if not METRICS_PATH.exists():
        return []
    with METRICS_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    numeric_fields = ["mae", "mse", "rmse", "r2", "class_accuracy", "class_macro_f1"]
    int_fields = ["horizon", "train_rows", "validation_rows", "test_rows", "feature_count"]
    for row in rows:
        for field in numeric_fields:
            if row.get(field):
                row[field] = float(row[field])
        for field in int_fields:
            if row.get(field):
                row[field] = int(row[field])
    return rows


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    models = load_best_models()
    return {
        "status": "ok",
        "app": APP_TITLE,
        "models": [
            {
                "horizon_hours": model.horizon,
                "model": model.model_name,
                "rmse": model.rmse,
                "path": str(model.model_path.relative_to(PROJECT_ROOT)),
            }
            for model in models.values()
        ],
    }


@app.get("/api/forecast")
def forecast(
    history_days: int = Query(
        DEFAULT_HISTORY_DAYS,
        ge=8,
        le=30,
        description="Number of past days requested from Open-Meteo. 8+ days are required for 168h rolling features.",
    )
) -> dict[str, Any]:
    try:
        frame = fetch_live_hanoi_frame(history_days)
        return forecast_from_live_frame(frame)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    try:
        return {
            "best_models": load_json(BEST_MODELS_PATH),
            "metrics": read_metrics(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
