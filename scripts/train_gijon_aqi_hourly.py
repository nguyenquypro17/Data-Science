from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX

try:
    from pyearth import Earth
except ImportError:  # pragma: no cover - optional dependency.
    Earth = None


RANDOM_STATE = 42

DEFAULT_DATA_PATH = Path("data/processed/hanoi_air_ml_ready_train.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/gijon_aqi_hourly")

POLLUTANT_COLS = ["pm25", "pm10", "no2", "o3", "co", "so2"]
TARGET_COL = "us_aqi"
BASE_SIGNAL_COLS = [TARGET_COL, *POLLUTANT_COLS]

WEATHER_COLS = [
    "temperature",
    "humidity",
    "dew_point",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "cloud_cover",
    "wind_speed",
    "wind_direction",
    "wind_gusts",
    "wind_u",
    "wind_v",
    "temp_humidity",
    "pressure_diff",
    "is_raining",
]

TIME_FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_dry_season",
]

LAGS = [1, 3, 6, 12, 24, 168]
ROLLING_WINDOWS = [3, 6, 24, 168]
WEATHER_LAGS = [1, 3, 6]


@dataclass
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_train_val: pd.DataFrame
    y_train_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    test_datetime_utc: pd.Series
    test_datetime_local: pd.Series
    test_row_id: pd.Series
    train_val_end_row_id: int
    test_start_row_id: int
    test_end_row_id: int


@dataclass
class ModelResult:
    horizon: int
    model_name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    class_accuracy: float
    class_macro_f1: float
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    model_path: str | None
    metadata_path: str | None
    prediction_path: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train hourly Hanoi AQI forecasting models following the spirit of "
            "the Gijon PM10 paper: ARIMA, VARMA, MLP, SVMR, and MARS."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    parser.add_argument("--target", default=TARGET_COL, choices=[TARGET_COL])
    parser.add_argument("--timezone", default="Asia/Bangkok")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--include-weather", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-time-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-arima", action="store_true")
    parser.add_argument("--skip-varma", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--skip-svmr", action="store_true")
    parser.add_argument("--skip-mars", action="store_true")
    parser.add_argument(
        "--allow-skip-mars-if-missing",
        action="store_true",
        help="Continue training other models when py-earth2/pyearth is not installed.",
    )
    parser.add_argument("--arima-max-p", type=int, default=3)
    parser.add_argument("--arima-max-d", type=int, default=1)
    parser.add_argument("--arima-max-q", type=int, default=3)
    parser.add_argument("--varma-max-p", type=int, default=2)
    parser.add_argument("--varma-max-q", type=int, default=1)
    parser.add_argument(
        "--varma-use-weather-exog",
        action="store_true",
        help=(
            "Use weather variables as exogenous variables in VARMAX. This assumes "
            "future weather values are available from a forecast service."
        ),
    )
    parser.add_argument(
        "--svmr-train-limit",
        type=int,
        default=0,
        help=(
            "Optional cap on SVMR training rows for slow machines. 0 means use all "
            "chronological train+validation rows."
        ),
    )
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}."
        )
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be positive.")


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "models": output_dir / "models",
        "reports": output_dir / "reports",
        "predictions": output_dir / "predictions",
        "metadata": output_dir / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_data(data_path: Path, timezone: str) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    required = ["datetime", TARGET_COL, *POLLUTANT_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert(timezone)
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)

    numeric_cols = [
        col
        for col in [TARGET_COL, *POLLUTANT_COLS, *WEATHER_COLS]
        if col in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["datetime_local"]
    hour = dt.dt.hour
    day_of_week = dt.dt.dayofweek
    month = dt.dt.month

    time_features = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7),
            "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_weekend": day_of_week.isin([5, 6]).astype(int),
            "is_dry_season": month.isin([11, 12, 1, 2, 3, 4]).astype(int),
        },
        index=out.index,
    )
    return pd.concat([out, time_features], axis=1).copy()


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feature_data: dict[str, pd.Series] = {}
    for col in BASE_SIGNAL_COLS:
        for lag in LAGS:
            feature_data[f"{col}_lag_{lag}h"] = out[col].shift(lag)

    for col in [TARGET_COL, "pm25", "pm10"]:
        for window in ROLLING_WINDOWS:
            rolling = out[col].rolling(window=window, min_periods=window)
            feature_data[f"{col}_roll_mean_{window}h"] = rolling.mean()
            feature_data[f"{col}_roll_std_{window}h"] = rolling.std()

    available_weather = [col for col in WEATHER_COLS if col in out.columns]
    for col in available_weather:
        for lag in WEATHER_LAGS:
            feature_data[f"{col}_lag_{lag}h"] = out[col].shift(lag)

    lag_features = pd.DataFrame(feature_data, index=out.index)
    return pd.concat([out, lag_features], axis=1).copy()


def build_feature_frame(
    df: pd.DataFrame,
    horizon: int,
    include_weather: bool,
    include_time_features: bool,
) -> tuple[pd.DataFrame, list[str]]:
    out = add_time_features(df)
    out = add_lag_and_rolling_features(out)
    out["row_id"] = np.arange(len(out))
    out[f"target_us_aqi_t+{horizon}"] = out[TARGET_COL].shift(-horizon)

    feature_cols: list[str] = []
    feature_cols.extend(BASE_SIGNAL_COLS)
    feature_cols.extend(
        col
        for col in out.columns
        if any(col.startswith(f"{base}_lag_") for base in BASE_SIGNAL_COLS)
    )
    feature_cols.extend(
        col
        for col in out.columns
        if any(col.startswith(f"{base}_roll_") for base in [TARGET_COL, "pm25", "pm10"])
    )

    if include_weather:
        weather_existing = [col for col in WEATHER_COLS if col in out.columns]
        feature_cols.extend(weather_existing)
        feature_cols.extend(
            col
            for col in out.columns
            if any(col.startswith(f"{base}_lag_") for base in weather_existing)
        )

    if include_time_features:
        feature_cols.extend([col for col in TIME_FEATURE_COLS if col in out.columns])

    feature_cols = sorted(dict.fromkeys(feature_cols))

    keep_cols = [
        "row_id",
        "datetime_utc",
        "datetime_local",
        TARGET_COL,
        f"target_us_aqi_t+{horizon}",
        *feature_cols,
    ]
    keep_cols = list(dict.fromkeys(keep_cols))
    out = out[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out, feature_cols


def chronological_split(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_ratio: float,
    val_ratio: float,
) -> SplitData:
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            f"Invalid split for {n} rows: train_end={train_end}, val_end={val_end}."
        )

    train = frame.iloc[:train_end]
    val = frame.iloc[train_end:val_end]
    train_val = frame.iloc[:val_end]
    test = frame.iloc[val_end:]

    return SplitData(
        x_train=train[feature_cols],
        y_train=train[target_col],
        x_val=val[feature_cols],
        y_val=val[target_col],
        x_train_val=train_val[feature_cols],
        y_train_val=train_val[target_col],
        x_test=test[feature_cols],
        y_test=test[target_col],
        test_datetime_utc=test["datetime_utc"],
        test_datetime_local=test["datetime_local"],
        test_row_id=test["row_id"],
        train_val_end_row_id=int(train_val["row_id"].iloc[-1]),
        test_start_row_id=int(test["row_id"].iloc[0]),
        test_end_row_id=int(test["row_id"].iloc[-1]),
    )


def aqi_category(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    bins = np.array([-np.inf, 50, 100, 150, 200, 300, np.inf])
    labels = np.array(
        [
            "good",
            "moderate",
            "unhealthy_sensitive",
            "unhealthy",
            "very_unhealthy",
            "hazardous",
        ]
    )
    return labels[np.digitize(arr, bins[1:-1], right=True)]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(y_pred, dtype=float)
    pred = np.clip(pred, 0, 500)
    true_class = aqi_category(y_true)
    pred_class = aqi_category(pred)
    mse = float(mean_squared_error(y_true, pred))

    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, pred)),
        "class_accuracy": float(accuracy_score(true_class, pred_class)),
        "class_macro_f1": float(f1_score(true_class, pred_class, average="macro")),
    }


def save_predictions(
    path: Path,
    horizon: int,
    model_name: str,
    split: SplitData,
    y_pred: np.ndarray,
) -> None:
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, 500)
    out = pd.DataFrame(
        {
            "datetime_utc": split.test_datetime_utc.astype(str).to_numpy(),
            "datetime_local": split.test_datetime_local.astype(str).to_numpy(),
            "horizon_hours": horizon,
            "model": model_name,
            "y_true_us_aqi": split.y_test.to_numpy(),
            "y_pred_us_aqi": pred,
            "true_aqi_class": aqi_category(split.y_test),
            "pred_aqi_class": aqi_category(pred),
            "abs_error": np.abs(split.y_test.to_numpy() - pred),
        }
    )
    out.to_csv(path, index=False)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def writable_float_vector(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return a writable C-contiguous target vector for Cython-backed estimators."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.array(arr, dtype=np.float64, order="C", copy=True)


def fit_best_sklearn_model(
    base_pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    split: SplitData,
    train_limit: int = 0,
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    x_train = split.x_train
    y_train = split.y_train
    if train_limit and len(x_train) > train_limit:
        x_train = x_train.iloc[-train_limit:]
        y_train = y_train.iloc[-train_limit:]

    best_model: Pipeline | None = None
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_rmse = np.inf

    for params in parameter_grid(param_grid):
        model = clone(base_pipeline)
        model.set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, writable_float_vector(y_train))
        pred_val = model.predict(split.x_val)
        metrics = regression_metrics(split.y_val, pred_val)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model = model
            best_params = params
            best_metrics = metrics

    if best_model is None or best_params is None or best_metrics is None:
        raise RuntimeError("No model was fitted from the sklearn grid.")

    final_model = clone(best_model)
    x_train_val = split.x_train_val
    y_train_val = split.y_train_val
    if train_limit and len(x_train_val) > train_limit:
        x_train_val = x_train_val.iloc[-train_limit:]
        y_train_val = y_train_val.iloc[-train_limit:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model.fit(x_train_val, writable_float_vector(y_train_val))

    return final_model, best_params, best_metrics


def train_persistence_baseline(
    horizon: int,
    split: SplitData,
    dirs: dict[str, Path],
) -> ModelResult:
    pred = split.x_test[TARGET_COL].to_numpy()
    metrics = regression_metrics(split.y_test, pred)
    pred_path = dirs["predictions"] / f"horizon_{horizon:02d}_baseline_persistence.csv"
    save_predictions(pred_path, horizon, "Baseline_Persistence", split, pred)

    return ModelResult(
        horizon=horizon,
        model_name="Baseline_Persistence",
        train_rows=len(split.x_train),
        validation_rows=len(split.x_val),
        test_rows=len(split.x_test),
        feature_count=1,
        model_path=None,
        metadata_path=None,
        prediction_path=str(pred_path),
        notes="Naive baseline: predicts future AQI as current AQI.",
        **metrics,
    )


def select_arima_order(
    series: pd.Series,
    max_p: int,
    max_d: int,
    max_q: int,
) -> tuple[int, int, int]:
    best_order: tuple[int, int, int] | None = None
    best_aic = np.inf
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and d == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(series, order=(p, d, q)).fit()
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception:
            continue
    if best_order is None:
        raise RuntimeError("Could not fit any ARIMA order. Try reducing or changing the grid.")
    return best_order


def rolling_arima_forecast(
    fit_result: Any,
    observed_test_series: pd.Series,
    horizon: int,
) -> np.ndarray:
    current_result = fit_result
    preds: list[float] = []
    for actual in observed_test_series:
        current_result = current_result.append([actual], refit=False)
        forecast = current_result.forecast(steps=horizon)
        preds.append(float(forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]))
    return np.asarray(preds)


def train_arima(
    horizon: int,
    raw_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    split: SplitData,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> ModelResult:
    history = raw_frame.loc[: split.train_val_end_row_id, TARGET_COL].astype(float).reset_index(drop=True)
    test_current = split.x_test[TARGET_COL].astype(float).reset_index(drop=True)

    order = select_arima_order(history, args.arima_max_p, args.arima_max_d, args.arima_max_q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(history, order=order).fit()

    pred = rolling_arima_forecast(model, test_current, horizon)
    metrics = regression_metrics(split.y_test, pred)

    model_path = dirs["models"] / f"horizon_{horizon:02d}_arima.pkl"
    metadata_path = dirs["metadata"] / f"horizon_{horizon:02d}_arima.json"
    pred_path = dirs["predictions"] / f"horizon_{horizon:02d}_arima.csv"

    save_pickle(model_path, model)
    save_json(
        metadata_path,
        {
            "model": "ARIMA",
            "horizon": horizon,
            "target": TARGET_COL,
            "order": order,
            "selection": "minimum AIC over configured p,d,q grid",
            "grid": {
                "max_p": args.arima_max_p,
                "max_d": args.arima_max_d,
                "max_q": args.arima_max_q,
            },
            "notes": "Univariate ARIMA using us_aqi only, evaluated with rolling-origin forecasts.",
        },
    )
    save_predictions(pred_path, horizon, "ARIMA", split, pred)

    return ModelResult(
        horizon=horizon,
        model_name="ARIMA",
        train_rows=len(split.x_train),
        validation_rows=len(split.x_val),
        test_rows=len(split.x_test),
        feature_count=1,
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        prediction_path=str(pred_path),
        notes=f"Selected ARIMA order {order}.",
        **metrics,
    )


def select_varma_order(
    endog: pd.DataFrame,
    exog: pd.DataFrame | None,
    max_p: int,
    max_q: int,
) -> tuple[int, int]:
    best_order: tuple[int, int] | None = None
    best_aic = np.inf
    for p, q in itertools.product(range(1, max_p + 1), range(max_q + 1)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VARMAX(
                    endog=endog,
                    exog=exog,
                    order=(p, q),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=200)
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, q)
        except Exception:
            continue
    if best_order is None:
        raise RuntimeError("Could not fit any VARMA/VARMAX order.")
    return best_order


def rolling_varma_forecast(
    fit_result: Any,
    test_endog_scaled: pd.DataFrame,
    horizon: int,
    target_position: int,
    endog_scaler: StandardScaler,
    test_exog_scaled: pd.DataFrame | None,
) -> np.ndarray:
    current_result = fit_result
    preds: list[float] = []
    for i in range(len(test_endog_scaled)):
        append_exog = None
        if test_exog_scaled is not None:
            append_exog = test_exog_scaled.iloc[[i]]
        current_result = current_result.append(
            test_endog_scaled.iloc[[i]],
            exog=append_exog,
            refit=False,
        )

        exog_future = None
        if test_exog_scaled is not None:
            future = test_exog_scaled.iloc[i + 1 : i + 1 + horizon]
            if len(future) < horizon:
                pad_source = future.iloc[[-1]] if len(future) else test_exog_scaled.iloc[[i]]
                padding = pd.concat([pad_source] * (horizon - len(future)))
                future = pd.concat([future, padding])
            exog_future = future

        forecast_scaled = current_result.forecast(steps=horizon, exog=exog_future)
        last_scaled = forecast_scaled.iloc[-1].to_numpy()
        last_unscaled = endog_scaler.inverse_transform(last_scaled.reshape(1, -1))[0]
        preds.append(float(last_unscaled[target_position]))
    return np.asarray(preds)


def train_varma(
    horizon: int,
    raw_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    split: SplitData,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> ModelResult:
    endog_cols = [TARGET_COL, *POLLUTANT_COLS]
    exog_cols = [col for col in WEATHER_COLS if col in raw_frame.columns] if args.varma_use_weather_exog else []

    train_val_end = split.train_val_end_row_id
    test_start = split.test_start_row_id
    test_end = split.test_end_row_id
    test_exog_end = min(len(raw_frame) - 1, test_end + horizon)

    train_val_raw = raw_frame.loc[:train_val_end, endog_cols + exog_cols].dropna()
    test_raw = raw_frame.loc[test_start:test_end, endog_cols + exog_cols].dropna()

    endog_scaler = StandardScaler()
    train_val_endog_scaled = pd.DataFrame(
        endog_scaler.fit_transform(train_val_raw[endog_cols]),
        columns=endog_cols,
        index=train_val_raw.index,
    )
    test_endog_scaled = pd.DataFrame(
        endog_scaler.transform(test_raw[endog_cols]),
        columns=endog_cols,
        index=test_raw.index,
    )

    exog_scaler: StandardScaler | None = None
    train_val_exog_scaled: pd.DataFrame | None = None
    test_exog_scaled: pd.DataFrame | None = None
    if exog_cols:
        exog_scaler = StandardScaler()
        train_val_exog_scaled = pd.DataFrame(
            exog_scaler.fit_transform(train_val_raw[exog_cols]),
            columns=exog_cols,
            index=train_val_raw.index,
        )
        test_exog_scaled = pd.DataFrame(
            exog_scaler.transform(raw_frame.loc[test_start:test_exog_end, exog_cols]),
            columns=exog_cols,
            index=raw_frame.loc[test_start:test_exog_end].index,
        )

    order = select_varma_order(
        train_val_endog_scaled,
        train_val_exog_scaled,
        args.varma_max_p,
        args.varma_max_q,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VARMAX(
            endog=train_val_endog_scaled,
            exog=train_val_exog_scaled,
            order=order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=300)

    pred = rolling_varma_forecast(
        model,
        test_endog_scaled=test_endog_scaled,
        horizon=horizon,
        target_position=endog_cols.index(TARGET_COL),
        endog_scaler=endog_scaler,
        test_exog_scaled=test_exog_scaled,
    )

    # Align to the split test length if VARMAX dropped rows due to missing values.
    if len(pred) != len(split.y_test):
        pred = pred[: len(split.y_test)]
        aligned_split = SplitData(
            x_train=split.x_train,
            y_train=split.y_train,
            x_val=split.x_val,
            y_val=split.y_val,
            x_train_val=split.x_train_val,
            y_train_val=split.y_train_val,
            x_test=split.x_test.iloc[: len(pred)],
            y_test=split.y_test.iloc[: len(pred)],
            test_datetime_utc=split.test_datetime_utc.iloc[: len(pred)],
            test_datetime_local=split.test_datetime_local.iloc[: len(pred)],
            test_row_id=split.test_row_id.iloc[: len(pred)],
            train_val_end_row_id=split.train_val_end_row_id,
            test_start_row_id=split.test_start_row_id,
            test_end_row_id=int(split.test_row_id.iloc[len(pred) - 1]),
        )
    else:
        aligned_split = split

    metrics = regression_metrics(aligned_split.y_test, pred)

    model_path = dirs["models"] / f"horizon_{horizon:02d}_varma.pkl"
    metadata_path = dirs["metadata"] / f"horizon_{horizon:02d}_varma.json"
    pred_path = dirs["predictions"] / f"horizon_{horizon:02d}_varma.csv"

    save_pickle(
        model_path,
        {
            "model": model,
            "endog_scaler": endog_scaler,
            "exog_scaler": exog_scaler,
            "endog_cols": endog_cols,
            "exog_cols": exog_cols,
            "order": order,
        },
    )
    save_json(
        metadata_path,
        {
            "model": "VARMA" if not exog_cols else "VARMAX",
            "horizon": horizon,
            "target": TARGET_COL,
            "order": order,
            "endog_cols": endog_cols,
            "exog_cols": exog_cols,
            "selection": "minimum AIC over configured p,q grid",
            "notes": (
                "Multivariate pollutant time-series model. Weather exog assumes "
                "future weather forecasts are available."
            ),
        },
    )
    save_predictions(pred_path, horizon, "VARMA", aligned_split, pred)

    return ModelResult(
        horizon=horizon,
        model_name="VARMA",
        train_rows=len(split.x_train),
        validation_rows=len(split.x_val),
        test_rows=len(aligned_split.x_test),
        feature_count=len(endog_cols) + len(exog_cols),
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        prediction_path=str(pred_path),
        notes=f"Selected VARMA/VARMAX order {order}.",
        **metrics,
    )


def train_mlp(
    horizon: int,
    split: SplitData,
    feature_cols: list[str],
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> ModelResult:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    random_state=RANDOM_STATE,
                    max_iter=args.mlp_max_iter,
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    grid = {
        "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "model__alpha": [0.0001, 0.001],
        "model__learning_rate_init": [0.001],
    }
    model, best_params, val_metrics = fit_best_sklearn_model(pipeline, grid, split)
    pred = model.predict(split.x_test)
    metrics = regression_metrics(split.y_test, pred)

    model_path = dirs["models"] / f"horizon_{horizon:02d}_mlp.joblib"
    metadata_path = dirs["metadata"] / f"horizon_{horizon:02d}_mlp.json"
    pred_path = dirs["predictions"] / f"horizon_{horizon:02d}_mlp.csv"

    joblib.dump(model, model_path)
    save_json(
        metadata_path,
        {
            "model": "MLP",
            "horizon": horizon,
            "target": TARGET_COL,
            "best_params": best_params,
            "validation_metrics": val_metrics,
            "feature_cols": feature_cols,
        },
    )
    save_predictions(pred_path, horizon, "MLP", split, pred)

    return ModelResult(
        horizon=horizon,
        model_name="MLP",
        train_rows=len(split.x_train),
        validation_rows=len(split.x_val),
        test_rows=len(split.x_test),
        feature_count=len(feature_cols),
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        prediction_path=str(pred_path),
        notes="Best MLP selected by validation RMSE, then refit on train+validation.",
        **metrics,
    )


def train_svmr(
    horizon: int,
    split: SplitData,
    feature_cols: list[str],
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> ModelResult:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR()),
        ]
    )
    grid = {
        "model__kernel": ["rbf"],
        "model__C": [1.0, 10.0, 100.0],
        "model__epsilon": [0.1, 1.0, 2.0],
        "model__gamma": ["scale"],
    }
    model, best_params, val_metrics = fit_best_sklearn_model(
        pipeline,
        grid,
        split,
        train_limit=args.svmr_train_limit,
    )
    pred = model.predict(split.x_test)
    metrics = regression_metrics(split.y_test, pred)

    model_path = dirs["models"] / f"horizon_{horizon:02d}_svmr.joblib"
    metadata_path = dirs["metadata"] / f"horizon_{horizon:02d}_svmr.json"
    pred_path = dirs["predictions"] / f"horizon_{horizon:02d}_svmr.csv"

    joblib.dump(model, model_path)
    save_json(
        metadata_path,
        {
            "model": "SVMR",
            "horizon": horizon,
            "target": TARGET_COL,
            "best_params": best_params,
            "validation_metrics": val_metrics,
            "feature_cols": feature_cols,
            "svmr_train_limit": args.svmr_train_limit,
        },
    )
    save_predictions(pred_path, horizon, "SVMR", split, pred)

    return ModelResult(
        horizon=horizon,
        model_name="SVMR",
        train_rows=len(split.x_train),
        validation_rows=len(split.x_val),
        test_rows=len(split.x_test),
        feature_count=len(feature_cols),
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        prediction_path=str(pred_path),
        notes="Best epsilon-SVR selected by validation RMSE, then refit on train+validation.",
        **metrics,
    )


def train_mars(
    horizon: int,
    split: SplitData,
    feature_cols: list[str],
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> ModelResult:
    if Earth is None:
        raise ImportError(
            "MARS requires py-earth2. Install it or run with "
            "--allow-skip-mars-if-missing to train the remaining models first."
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Earth()),
        ]
    )
    grid = {
        "model__max_terms": [30, 60, 90],
        "model__max_degree": [1, 2],
        "model__penalty": [2.0, 3.0],
    }
    model, best_params, val_metrics = fit_best_sklearn_model(pipeline, grid, split)
    pred = model.predict(split.x_test)
    metrics = regression_metrics(split.y_test, pred)

    model_path = dirs["models"] / f"horizon_{horizon:02d}_mars.joblib"
    metadata_path = dirs["metadata"] / f"horizon_{horizon:02d}_mars.json"
    pred_path = dirs["predictions"] / f"horizon_{horizon:02d}_mars.csv"

    joblib.dump(model, model_path)
    save_json(
        metadata_path,
        {
            "model": "MARS",
            "horizon": horizon,
            "target": TARGET_COL,
            "best_params": best_params,
            "validation_metrics": val_metrics,
            "feature_cols": feature_cols,
        },
    )
    save_predictions(pred_path, horizon, "MARS", split, pred)

    return ModelResult(
        horizon=horizon,
        model_name="MARS",
        train_rows=len(split.x_train),
        validation_rows=len(split.x_val),
        test_rows=len(split.x_test),
        feature_count=len(feature_cols),
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        prediction_path=str(pred_path),
        notes="Best MARS/Earth model selected by validation RMSE, then refit on train+validation.",
        **metrics,
    )


def pick_best_models(results: list[ModelResult]) -> dict[str, Any]:
    by_horizon: dict[int, list[ModelResult]] = {}
    for result in results:
        if result.model_name.startswith("Baseline"):
            continue
        by_horizon.setdefault(result.horizon, []).append(result)

    best: dict[str, Any] = {}
    for horizon, items in by_horizon.items():
        winner = min(items, key=lambda item: item.rmse)
        best[str(horizon)] = {
            "model_name": winner.model_name,
            "rmse": winner.rmse,
            "mae": winner.mae,
            "r2": winner.r2,
            "model_path": winner.model_path,
            "metadata_path": winner.metadata_path,
        }
    return best


def train_for_horizon(
    horizon: int,
    df: pd.DataFrame,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> list[ModelResult]:
    print(f"\n=== Horizon t+{horizon}h ===")
    frame, feature_cols = build_feature_frame(
        df,
        horizon=horizon,
        include_weather=args.include_weather,
        include_time_features=args.include_time_features,
    )
    target_col = f"target_us_aqi_t+{horizon}"
    split = chronological_split(
        frame,
        feature_cols=feature_cols,
        target_col=target_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    feature_path = dirs["metadata"] / f"horizon_{horizon:02d}_feature_columns.json"
    save_json(
        feature_path,
        {
            "horizon": horizon,
            "target_col": target_col,
            "feature_cols": feature_cols,
            "rows_after_feature_engineering": len(frame),
            "split": {
                "train_rows": len(split.x_train),
                "validation_rows": len(split.x_val),
                "test_rows": len(split.x_test),
            },
        },
    )

    results: list[ModelResult] = []

    if args.include_baseline:
        results.append(train_persistence_baseline(horizon, split, dirs))

    if not args.skip_arima:
        print("Training ARIMA...")
        results.append(train_arima(horizon, df, frame, split, dirs, args))

    if not args.skip_varma:
        print("Training VARMA/VARMAX...")
        results.append(train_varma(horizon, df, frame, split, dirs, args))

    if not args.skip_mlp:
        print("Training MLP...")
        results.append(train_mlp(horizon, split, feature_cols, dirs, args))

    if not args.skip_svmr:
        print("Training SVMR...")
        results.append(train_svmr(horizon, split, feature_cols, dirs, args))

    if not args.skip_mars:
        print("Training MARS...")
        if Earth is None and args.allow_skip_mars_if_missing:
            print("MARS skipped because py-earth2/pyearth is not installed.")
        else:
            results.append(train_mars(horizon, split, feature_cols, dirs, args))

    return results


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    dirs = ensure_dirs(args.output_dir)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    save_json(dirs["metadata"] / "run_config.json", config)

    df = load_data(args.data, args.timezone)
    all_results: list[ModelResult] = []

    for horizon in args.horizons:
        all_results.extend(train_for_horizon(horizon, df, dirs, args))

    metrics_df = pd.DataFrame([asdict(result) for result in all_results])
    metrics_df = metrics_df.sort_values(["horizon", "rmse", "model_name"]).reset_index(drop=True)
    metrics_path = dirs["reports"] / "metrics_by_horizon.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_models = pick_best_models(all_results)
    save_json(dirs["reports"] / "best_models.json", best_models)

    print("\nTraining complete.")
    print(f"Metrics: {metrics_path}")
    print(f"Best models: {dirs['reports'] / 'best_models.json'}")


if __name__ == "__main__":
    main()
