from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.monitoring.drift_injector import DriftScenario, create_drift_scenario
from src.monitoring.drift_report import run_evidently_feature_drift
from src.monitoring.psi_detector import detect_psi_drift


def _load_base_dataset(
    config_path: str | Path,
) -> tuple[pd.DataFrame, str]:
    config = load_config(config_path)
    target = config["raw_data_config"]["target"]
    stream_path = resolve_path(
        config["external_data_config"]["external_data_csv"],
        config_path,
    )

    base_df = pd.read_csv(stream_path)
    return base_df, target


def _feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    return [column for column in df.columns if column != target]


def prepare_reference_and_stream(
    base_df: pd.DataFrame,
    *,
    target: str,
    random_state: int,
    reference_fraction: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = _feature_columns(base_df, target)
    shuffled = base_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    features = shuffled[feature_columns].copy()

    minimum_stream_rows = 1000
    reference_rows = int(round(len(features) * reference_fraction))
    reference_rows = min(reference_rows, len(features) - minimum_stream_rows)
    reference_rows = max(reference_rows, 1000)

    reference_df = features.iloc[:reference_rows].reset_index(drop=True)
    stream_df = features.iloc[reference_rows:].reset_index(drop=True)
    return reference_df, stream_df


def build_stream_with_drift(
    base_stream: pd.DataFrame,
    *,
    scenario: str,
    magnitude: float,
    drift_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, DriftScenario, int]:
    drift_count = max(1, int(round(len(base_stream) * drift_fraction)))
    true_drift_start = max(0, len(base_stream) - drift_count)

    clean_segment = base_stream.iloc[:true_drift_start].copy()
    drift_segment = base_stream.iloc[true_drift_start:].copy()
    drifted_segment, metadata = create_drift_scenario(
        drift_segment,
        scenario=scenario,
        magnitude=magnitude,
        random_state=random_state,
    )
    stream = pd.concat([clean_segment, drifted_segment], ignore_index=True)
    return stream, metadata, true_drift_start


def _batch_ranges(total_rows: int, batch_size: int) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    for batch_index, start in enumerate(range(0, total_rows, batch_size), start=1):
        end = min(start + batch_size, total_rows)
        ranges.append((batch_index, start, end))
    return ranges


def _latency_records(true_drift_start: int, detected_end_row: int | None) -> float | None:
    if detected_end_row is None:
        return None
    return float(detected_end_row - true_drift_start)


def _build_evidently_detector(
    feature_name: str,
) -> Callable[[pd.Series, pd.Series], dict[str, Any]]:
    def detector(expected: pd.Series, actual: pd.Series) -> dict[str, Any]:
        return run_evidently_feature_drift(
            reference_data=pd.DataFrame({feature_name: expected}),
            current_data=pd.DataFrame({feature_name: actual}),
            feature_name=feature_name,
        )

    return detector


def _build_psi_detector() -> Callable[[pd.Series, pd.Series], dict[str, Any]]:
    def detector(expected: pd.Series, actual: pd.Series) -> dict[str, Any]:
        return {
            **detect_psi_drift(expected, actual),
            "p_value": np.nan,
        }

    return detector


def evaluate_detector_over_stream(
    detector_name: str,
    detector: Callable[[pd.Series, pd.Series], dict[str, Any]],
    reference_df: pd.DataFrame,
    stream_df: pd.DataFrame,
    *,
    feature_name: str,
    scenario: str,
    true_drift_start: int,
    magnitude: float,
    batch_size: int,
    min_consecutive_detections: int = 3,
) -> dict[str, Any]:
    detected_at_batch: int | None = None
    detected_end_row: int | None = None
    p_value: float | None = None
    drift_score: float | None = None
    consecutive_hits = 0

    for batch_index, start, end in _batch_ranges(len(stream_df), batch_size):
        del start
        current_window = stream_df.iloc[:end]
        result = detector(reference_df[feature_name], current_window[feature_name])
        drift_score = float(result["drift_score"])
        p_value = result.get("p_value")
        if bool(result["detected"]):
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        if consecutive_hits >= min_consecutive_detections:
            detected_at_batch = batch_index
            detected_end_row = end
            break

    return {
        "detector": detector_name,
        "scenario": scenario,
        "feature_drifted": feature_name,
        "magnitude": float(magnitude),
        "true_drift_start": int(true_drift_start),
        "detected_at_batch": detected_at_batch,
        "latency_records": _latency_records(true_drift_start, detected_end_row),
        "p_value": float(p_value) if p_value is not None else np.nan,
        "drift_score": float(drift_score) if drift_score is not None else np.nan,
        "detected": bool(detected_at_batch is not None),
    }


def evaluate_drift_scenarios(
    config_path: str | Path = "params.yaml",
    *,
    scenarios: list[str] | None = None,
    magnitude: float = 0.2,
    batch_size: int = 500,
    drift_fraction: float = 0.2,
    random_state: int = 111,
) -> pd.DataFrame:
    base_df, target = _load_base_dataset(config_path)
    scenario_list = scenarios or ["gradual", "sudden", "seasonal"]

    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenario_list):
        reference_features, clean_stream_features = prepare_reference_and_stream(
            base_df,
            target=target,
            random_state=random_state + scenario_index,
        )
        stream, metadata, true_drift_start = build_stream_with_drift(
            clean_stream_features,
            scenario=scenario,
            magnitude=magnitude,
            drift_fraction=drift_fraction,
            random_state=random_state + scenario_index,
        )
        feature_name = metadata.feature_drifted

        for detector_name in ["evidently", "psi"]:
            if detector_name == "evidently":
                detector = _build_evidently_detector(feature_name)
            else:
                detector = _build_psi_detector()

            rows.append(
                evaluate_detector_over_stream(
                    detector_name,
                    detector,
                    reference_features,
                    stream,
                    feature_name=feature_name,
                    scenario=scenario,
                    true_drift_start=true_drift_start,
                    magnitude=magnitude,
                    batch_size=batch_size,
                )
            )

    return pd.DataFrame(rows)


def run_clean_false_positive_checks(
    config_path: str | Path,
    *,
    batch_size: int,
    repeats: int = 3,
    random_state: int = 111,
) -> pd.DataFrame:
    base_df, target = _load_base_dataset(config_path)
    feature_name = "total_day_minutes"
    rows: list[dict[str, Any]] = []

    for repeat_index in range(repeats):
        reference_features, stream = prepare_reference_and_stream(
            base_df,
            target=target,
            random_state=random_state + repeat_index,
        )
        for detector_name in ["evidently", "psi"]:
            if detector_name == "evidently":
                detector = _build_evidently_detector(feature_name)
            else:
                detector = _build_psi_detector()

            rows.append(
                evaluate_detector_over_stream(
                    detector_name,
                    detector,
                    reference_features,
                    stream,
                    feature_name=feature_name,
                    scenario="clean",
                    true_drift_start=len(stream) + 1,
                    magnitude=0.0,
                    batch_size=batch_size,
                )
            )

    return pd.DataFrame(rows)


def build_sensitivity_summary(
    config_path: str | Path,
    *,
    magnitudes: list[float],
    batch_size: int,
    drift_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    clean_df = run_clean_false_positive_checks(
        config_path,
        batch_size=batch_size,
        repeats=3,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []

    for magnitude in magnitudes:
        evaluation_df = evaluate_drift_scenarios(
            config_path,
            magnitude=magnitude,
            batch_size=batch_size,
            drift_fraction=drift_fraction,
            random_state=random_state,
        )
        for detector_name in ["evidently", "psi"]:
            detector_rows = evaluation_df.loc[evaluation_df["detector"] == detector_name].copy()
            clean_rows = clean_df.loc[clean_df["detector"] == detector_name].copy()

            true_detection_mask = detector_rows["detected"] & (
                detector_rows["latency_records"].fillna(0) >= 0
            )
            mean_latency = detector_rows.loc[true_detection_mask, "latency_records"].mean()

            rows.append(
                {
                    "detector": detector_name,
                    "magnitude": magnitude,
                    "detection_rate": float(true_detection_mask.mean()),
                    "false_positive_rate": float(clean_rows["detected"].mean()),
                    "mean_latency": (
                        float(mean_latency) if not math.isnan(mean_latency) else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def _draw_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    fmt: str,
) -> None:
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            label = "n/a" if np.isnan(value) else format(value, fmt)
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_sensitivity_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    detectors = ["evidently", "psi"]
    magnitudes = sorted(summary_df["magnitude"].unique().tolist())
    magnitude_labels = [f"{int(magnitude * 100)}%" for magnitude in magnitudes]

    detection_matrix = np.array(
        [
            [
                summary_df.loc[
                    (summary_df["detector"] == detector) & (summary_df["magnitude"] == magnitude),
                    "detection_rate",
                ].iloc[0]
                for magnitude in magnitudes
            ]
            for detector in detectors
        ]
    )
    false_positive_matrix = np.array(
        [
            [
                summary_df.loc[
                    (summary_df["detector"] == detector) & (summary_df["magnitude"] == magnitude),
                    "false_positive_rate",
                ].iloc[0]
                for magnitude in magnitudes
            ]
            for detector in detectors
        ]
    )
    latency_matrix = np.array(
        [
            [
                summary_df.loc[
                    (summary_df["detector"] == detector) & (summary_df["magnitude"] == magnitude),
                    "mean_latency",
                ].iloc[0]
                for magnitude in magnitudes
            ]
            for detector in detectors
        ]
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    detector_labels = [detector.title() for detector in detectors]
    _draw_heatmap(
        axes[0],
        detection_matrix,
        detector_labels,
        magnitude_labels,
        "Detection Rate",
        ".2f",
    )
    _draw_heatmap(
        axes[1],
        false_positive_matrix,
        detector_labels,
        magnitude_labels,
        "False Positive Rate",
        ".2f",
    )
    _draw_heatmap(
        axes[2],
        latency_matrix,
        detector_labels,
        magnitude_labels,
        "Mean Latency (records)",
        ".0f",
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_drift_monitoring_evaluation(
    config_path: str | Path = "params.yaml",
    *,
    results_path: str | Path = "results/drift_evaluation.csv",
    heatmap_path: str | Path = "plots/drift_sensitivity_heatmap.png",
    batch_size: int = 500,
    drift_fraction: float = 0.2,
    baseline_magnitude: float = 0.2,
    random_state: int = 111,
) -> dict[str, Any]:
    evaluation_df = evaluate_drift_scenarios(
        config_path,
        magnitude=baseline_magnitude,
        batch_size=batch_size,
        drift_fraction=drift_fraction,
        random_state=random_state,
    )

    results_output_path = resolve_path(results_path, config_path)
    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(results_output_path, index=False)

    sensitivity_df = build_sensitivity_summary(
        config_path,
        magnitudes=[0.05, 0.10, 0.20, 0.40],
        batch_size=batch_size,
        drift_fraction=drift_fraction,
        random_state=random_state,
    )
    heatmap_output_path = resolve_path(heatmap_path, config_path)
    save_sensitivity_heatmap(sensitivity_df, heatmap_output_path)

    return {
        "results_path": str(results_output_path),
        "heatmap_path": str(heatmap_output_path),
        "evaluation_rows": int(len(evaluation_df)),
        "sensitivity_rows": int(len(sensitivity_df)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--results-path", default="results/drift_evaluation.csv")
    parser.add_argument("--heatmap-path", default="plots/drift_sensitivity_heatmap.png")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--drift-fraction", type=float, default=0.2)
    parser.add_argument("--baseline-magnitude", type=float, default=0.2)
    args = parser.parse_args()

    run_drift_monitoring_evaluation(
        config_path=args.config,
        results_path=args.results_path,
        heatmap_path=args.heatmap_path,
        batch_size=args.batch_size,
        drift_fraction=args.drift_fraction,
        baseline_magnitude=args.baseline_magnitude,
    )
