from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/gijon_aqi_hourly")
MODEL_ORDER = [
    "Baseline_Persistence",
    "ARIMA",
    "VARMA",
    "VARMAX",
    "MLP",
    "SVMR",
    "MARS",
]
AQI_CLASSES = [
    "good",
    "moderate",
    "unhealthy_sensitive",
    "unhealthy",
    "very_unhealthy",
    "hazardous",
]
AQI_THRESHOLDS = [50, 100, 150, 200, 300]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Gijon-style AQI forecasting results with matplotlib."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--zoom-hours", type=int, default=168)
    parser.add_argument("--scatter-sample", type=int, default=2500)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    normalized = Path(str(path_value).replace("\\", "/"))
    return normalized


def ensure_fig_dir(output_dir: Path, fig_dir: Path | None) -> Path:
    out = fig_dir if fig_dir is not None else output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def sort_models(models: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda name: (order.get(name, 999), name))


def load_metrics(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "reports" / "metrics_by_horizon.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    metrics = pd.read_csv(path)
    numeric_cols = [
        "horizon",
        "mae",
        "mse",
        "rmse",
        "r2",
        "class_accuracy",
        "class_macro_f1",
    ]
    for col in numeric_cols:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    return metrics


def load_best_model_names(output_dir: Path, metrics: pd.DataFrame) -> dict[int, str]:
    path = output_dir / "reports" / "best_models.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(horizon): item["model_name"] for horizon, item in data.items()}

    best: dict[int, str] = {}
    for horizon, group in metrics.groupby("horizon"):
        group = group[~group["model_name"].str.startswith("Baseline")]
        if len(group):
            best[int(horizon)] = group.sort_values("rmse").iloc[0]["model_name"]
    return best


def load_prediction(path_value: str | Path) -> pd.DataFrame:
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    df["datetime_local"] = (
        pd.to_datetime(df["datetime_local"], utc=True)
        .dt.tz_convert("Asia/Bangkok")
        .dt.tz_localize(None)
    )
    df["error"] = df["y_pred_us_aqi"] - df["y_true_us_aqi"]
    df["abs_error"] = df["error"].abs()
    return df


def load_all_predictions(metrics: pd.DataFrame) -> dict[tuple[int, str], pd.DataFrame]:
    predictions: dict[tuple[int, str], pd.DataFrame] = {}
    for row in metrics.itertuples(index=False):
        path = getattr(row, "prediction_path")
        if isinstance(path, str) and path:
            key = (int(row.horizon), str(row.model_name))
            predictions[key] = load_prediction(path)
    return predictions


def add_aqi_threshold_lines(ax: plt.Axes) -> None:
    for threshold in AQI_THRESHOLDS:
        ax.axhline(threshold, color="0.75", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_ylim(bottom=0)


def save_current(fig_dir: Path, filename: str, dpi: int) -> Path:
    path = fig_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_metric_comparison(metrics: pd.DataFrame, fig_dir: Path, dpi: int) -> Path:
    plot_metrics = [
        ("mae", "MAE lower is better"),
        ("rmse", "RMSE lower is better"),
        ("r2", "R2 higher is better"),
        ("class_macro_f1", "Macro-F1 higher is better"),
    ]
    horizons = sorted(metrics["horizon"].unique())
    models = sort_models(metrics["model_name"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.ravel()
    x = np.arange(len(models))
    width = 0.8 / max(len(horizons), 1)

    for ax, (metric, title) in zip(axes, plot_metrics):
        for i, horizon in enumerate(horizons):
            values = []
            group = metrics[metrics["horizon"] == horizon].set_index("model_name")
            for model in models:
                values.append(group.loc[model, metric] if model in group.index else np.nan)
            offset = (i - (len(horizons) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=f"t+{int(horizon)}h")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    return save_current(fig_dir, "01_metric_comparison_by_model.png", dpi)


def plot_baseline_improvement(metrics: pd.DataFrame, fig_dir: Path, dpi: int) -> Path:
    rows = []
    for horizon, group in metrics.groupby("horizon"):
        baseline = group[group["model_name"] == "Baseline_Persistence"]
        if baseline.empty:
            continue
        baseline_rmse = float(baseline.iloc[0]["rmse"])
        baseline_mae = float(baseline.iloc[0]["mae"])
        for row in group.itertuples(index=False):
            if row.model_name == "Baseline_Persistence":
                continue
            rows.append(
                {
                    "horizon": int(horizon),
                    "model_name": row.model_name,
                    "rmse_improvement": (baseline_rmse - float(row.rmse)) / baseline_rmse * 100,
                    "mae_improvement": (baseline_mae - float(row.mae)) / baseline_mae * 100,
                }
            )
    imp = pd.DataFrame(rows)
    if imp.empty:
        raise ValueError("No baseline rows found for improvement plot.")

    horizons = sorted(imp["horizon"].unique())
    models = sort_models(imp["model_name"].unique().tolist())
    x = np.arange(len(models))
    width = 0.8 / max(len(horizons), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for ax, metric, title in [
        (axes[0], "rmse_improvement", "RMSE improvement vs persistence baseline"),
        (axes[1], "mae_improvement", "MAE improvement vs persistence baseline"),
    ]:
        for i, horizon in enumerate(horizons):
            group = imp[imp["horizon"] == horizon].set_index("model_name")
            values = [group.loc[m, metric] if m in group.index else np.nan for m in models]
            offset = (i - (len(horizons) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=f"t+{horizon}h")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("% improvement")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    return save_current(fig_dir, "02_improvement_vs_baseline.png", dpi)


def best_metric_row(metrics: pd.DataFrame, horizon: int, model_name: str) -> pd.Series:
    row = metrics[(metrics["horizon"] == horizon) & (metrics["model_name"] == model_name)]
    if row.empty:
        raise ValueError(f"Missing metrics for horizon={horizon}, model={model_name}")
    return row.iloc[0]


def plot_best_timeseries(
    metrics: pd.DataFrame,
    predictions: dict[tuple[int, str], pd.DataFrame],
    best_models: dict[int, str],
    fig_dir: Path,
    dpi: int,
    zoom_hours: int,
) -> tuple[Path, Path]:
    horizons = sorted(best_models)
    fig, axes = plt.subplots(len(horizons), 1, figsize=(16, 4.2 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        model = best_models[horizon]
        pred = predictions[(horizon, model)]
        metric_row = best_metric_row(metrics, horizon, model)
        ax.plot(pred["datetime_local"], pred["y_true_us_aqi"], label="True AQI", linewidth=1.2)
        ax.plot(pred["datetime_local"], pred["y_pred_us_aqi"], label=f"Predicted AQI ({model})", linewidth=1.0)
        add_aqi_threshold_lines(ax)
        ax.set_title(
            f"Best t+{horizon}h: {model} | MAE={metric_row['mae']:.2f}, "
            f"RMSE={metric_row['rmse']:.2f}, R2={metric_row['r2']:.3f}"
        )
        ax.set_ylabel("US AQI")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
    full_path = save_current(fig_dir, "03_best_models_true_vs_pred_full_test.png", dpi)

    fig, axes = plt.subplots(len(horizons), 1, figsize=(16, 4.2 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        model = best_models[horizon]
        pred = predictions[(horizon, model)].head(zoom_hours)
        ax.plot(pred["datetime_local"], pred["y_true_us_aqi"], label="True AQI", marker="o", markersize=2, linewidth=1.1)
        ax.plot(pred["datetime_local"], pred["y_pred_us_aqi"], label=f"Predicted AQI ({model})", marker="o", markersize=2, linewidth=1.0)
        add_aqi_threshold_lines(ax)
        ax.set_title(f"Zoom first {zoom_hours} test hours | t+{horizon}h | {model}")
        ax.set_ylabel("US AQI")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
    zoom_path = save_current(fig_dir, "04_best_models_true_vs_pred_zoom.png", dpi)
    return full_path, zoom_path


def plot_best_scatter(
    metrics: pd.DataFrame,
    predictions: dict[tuple[int, str], pd.DataFrame],
    best_models: dict[int, str],
    fig_dir: Path,
    dpi: int,
    sample_size: int,
) -> Path:
    horizons = sorted(best_models)
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.3 * len(horizons), 5.2), sharex=False, sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        model = best_models[horizon]
        pred = predictions[(horizon, model)]
        if len(pred) > sample_size:
            pred = pred.sample(sample_size, random_state=42).sort_index()
        metric_row = best_metric_row(metrics, horizon, model)
        x = pred["y_true_us_aqi"].to_numpy()
        y = pred["y_pred_us_aqi"].to_numpy()
        ax.scatter(x, y, s=10, alpha=0.35)
        lo = max(0, min(x.min(), y.min()) - 5)
        hi = max(x.max(), y.max()) + 5
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, label="Perfect prediction")
        for threshold in AQI_THRESHOLDS:
            ax.axvline(threshold, color="0.85", linewidth=0.6, linestyle="--")
            ax.axhline(threshold, color="0.85", linewidth=0.6, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"t+{horizon}h {model}\nRMSE={metric_row['rmse']:.2f}, R2={metric_row['r2']:.3f}")
        ax.set_xlabel("True US AQI")
        ax.set_ylabel("Predicted US AQI")
        ax.grid(alpha=0.2)
    return save_current(fig_dir, "05_best_models_scatter_true_vs_pred.png", dpi)


def plot_error_distributions(
    predictions: dict[tuple[int, str], pd.DataFrame],
    fig_dir: Path,
    dpi: int,
) -> Path:
    horizons = sorted({h for h, _ in predictions})
    fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 4.5 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        models = sort_models([m for h, m in predictions if h == horizon])
        data = [predictions[(horizon, model)]["abs_error"].to_numpy() for model in models]
        ax.boxplot(data, labels=models, showfliers=False)
        ax.set_title(f"Absolute error distribution | t+{horizon}h")
        ax.set_ylabel("Absolute error in AQI points")
        ax.grid(axis="y", alpha=0.25)
    return save_current(fig_dir, "06_abs_error_boxplots_by_model.png", dpi)


def plot_best_residual_histograms(
    predictions: dict[tuple[int, str], pd.DataFrame],
    best_models: dict[int, str],
    fig_dir: Path,
    dpi: int,
) -> Path:
    horizons = sorted(best_models)
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.3 * len(horizons), 4.8))
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        model = best_models[horizon]
        pred = predictions[(horizon, model)]
        residual = pred["error"].to_numpy()
        bias = residual.mean()
        ax.hist(residual, bins=45, color="#4C78A8", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.0, label="Zero error")
        ax.axvline(bias, color="#D62728", linewidth=1.2, linestyle="--", label=f"Bias={bias:.2f}")
        ax.set_title(f"Residuals | t+{horizon}h | {model}")
        ax.set_xlabel("Prediction error: pred - true")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    return save_current(fig_dir, "07_best_models_residual_histograms.png", dpi)


def confusion_matrix_counts(df: pd.DataFrame) -> np.ndarray:
    index = {name: i for i, name in enumerate(AQI_CLASSES)}
    matrix = np.zeros((len(AQI_CLASSES), len(AQI_CLASSES)), dtype=int)
    for true_label, pred_label in zip(df["true_aqi_class"], df["pred_aqi_class"]):
        if true_label in index and pred_label in index:
            matrix[index[true_label], index[pred_label]] += 1
    return matrix


def plot_best_confusion_matrices(
    predictions: dict[tuple[int, str], pd.DataFrame],
    best_models: dict[int, str],
    fig_dir: Path,
    dpi: int,
) -> Path:
    horizons = sorted(best_models)
    fig, axes = plt.subplots(1, len(horizons), figsize=(6.1 * len(horizons), 5.7))
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        model = best_models[horizon]
        matrix = confusion_matrix_counts(predictions[(horizon, model)])
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Class confusion | t+{horizon}h | {model}")
        ax.set_xticks(np.arange(len(AQI_CLASSES)))
        ax.set_yticks(np.arange(len(AQI_CLASSES)))
        ax.set_xticklabels(AQI_CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(AQI_CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for i in range(len(AQI_CLASSES)):
            for j in range(len(AQI_CLASSES)):
                if matrix[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        f"{norm[i, j] * 100:.0f}%\n{matrix[i, j]}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black" if norm[i, j] < 0.6 else "white",
                    )
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    return save_current(fig_dir, "08_best_models_confusion_matrices.png", dpi)


def write_worst_errors(
    predictions: dict[tuple[int, str], pd.DataFrame],
    best_models: dict[int, str],
    fig_dir: Path,
    n: int = 20,
) -> Path:
    rows = []
    for horizon, model in sorted(best_models.items()):
        pred = predictions[(horizon, model)].sort_values("abs_error", ascending=False).head(n).copy()
        pred.insert(0, "best_model", model)
        pred.insert(0, "horizon", horizon)
        rows.append(pred)
    out = pd.concat(rows, ignore_index=True)
    path = fig_dir / "worst_errors_best_models.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    fig_dir = ensure_fig_dir(args.output_dir, args.fig_dir)
    metrics = load_metrics(args.output_dir)
    best_models = load_best_model_names(args.output_dir, metrics)
    predictions = load_all_predictions(metrics)

    needed = [(horizon, model) for horizon, model in best_models.items()]
    missing = [key for key in needed if key not in predictions]
    if missing:
        raise FileNotFoundError(f"Missing prediction files for best models: {missing}")

    generated = [
        plot_metric_comparison(metrics, fig_dir, args.dpi),
        plot_baseline_improvement(metrics, fig_dir, args.dpi),
        *plot_best_timeseries(metrics, predictions, best_models, fig_dir, args.dpi, args.zoom_hours),
        plot_best_scatter(metrics, predictions, best_models, fig_dir, args.dpi, args.scatter_sample),
        plot_error_distributions(predictions, fig_dir, args.dpi),
        plot_best_residual_histograms(predictions, best_models, fig_dir, args.dpi),
        plot_best_confusion_matrices(predictions, best_models, fig_dir, args.dpi),
        write_worst_errors(predictions, best_models, fig_dir),
    ]

    print("Generated result figures/tables:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
