from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.models.predict import ChurnPredictor


class EvidentlyDriftError(RuntimeError):
    """Raised when Evidently cannot be imported or executed for drift monitoring."""


@dataclass(frozen=True)
class EvidentlyAPI:
    report_cls: type
    data_drift_preset_cls: type
    value_drift_cls: type | None
    column_drift_metric_cls: type | None


def _load_evidently_api() -> EvidentlyAPI:
    errors: list[str] = []

    try:
        evidently_module = importlib.import_module("evidently")
        report_cls = getattr(evidently_module, "Report")
        presets_module = importlib.import_module("evidently.presets")
        metrics_module = importlib.import_module("evidently.metrics")
        return EvidentlyAPI(
            report_cls=report_cls,
            data_drift_preset_cls=getattr(presets_module, "DataDriftPreset"),
            value_drift_cls=getattr(metrics_module, "ValueDrift", None),
            column_drift_metric_cls=getattr(metrics_module, "ColumnDriftMetric", None),
        )
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"modern API import failed: {exc}")

    try:
        report_module = importlib.import_module("evidently.report")
        preset_module = importlib.import_module("evidently.metric_preset")
        metrics_module = importlib.import_module("evidently.metrics")
        return EvidentlyAPI(
            report_cls=getattr(report_module, "Report"),
            data_drift_preset_cls=getattr(preset_module, "DataDriftPreset"),
            value_drift_cls=getattr(metrics_module, "ValueDrift", None),
            column_drift_metric_cls=getattr(metrics_module, "ColumnDriftMetric", None),
        )
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"legacy API import failed: {exc}")

    raise EvidentlyDriftError(
        "Evidently drift monitoring could not be initialized. "
        "Install a compatible Evidently runtime and rerun the job. "
        + " | ".join(errors)
    )


def _create_report(api: EvidentlyAPI, metrics: list[Any]):
    report_signature = inspect.signature(api.report_cls)
    if "metrics" in report_signature.parameters:
        return api.report_cls(metrics=metrics)
    return api.report_cls(metrics)


def _run_report(report, reference_data: pd.DataFrame, current_data: pd.DataFrame):
    run_signature = inspect.signature(report.run)
    if "reference_data" in run_signature.parameters:
        return report.run(reference_data=reference_data, current_data=current_data)
    return report.run(reference_data, current_data)


def _snapshot_to_dict(snapshot, report) -> dict[str, Any]:
    if snapshot is not None:
        if hasattr(snapshot, "dict"):
            return snapshot.dict()
        if hasattr(snapshot, "dump_dict"):
            return snapshot.dump_dict()
    if hasattr(report, "as_dict"):
        return report.as_dict()
    raise EvidentlyDriftError("Unable to serialize Evidently report output.")


def _save_snapshot_html(snapshot, report_path: Path, report) -> None:
    if snapshot is not None and hasattr(snapshot, "save_html"):
        snapshot.save_html(str(report_path))
        return
    if hasattr(report, "save_html"):
        report.save_html(str(report_path))
        return
    raise EvidentlyDriftError("Unable to save Evidently HTML report.")


def _find_metric_entry(report_dict: dict[str, Any], metric_types: set[str], column: str | None) -> dict[str, Any] | None:
    for metric in report_dict.get("metrics", []):
        config = metric.get("config", {})
        metric_type = str(config.get("type", ""))
        metric_column = config.get("column")
        if metric_type in metric_types and (column is None or metric_column == column):
            return metric
    return None


def run_evidently_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    *,
    feature_name: str,
    threshold: float = 0.05,
) -> dict[str, Any]:
    api = _load_evidently_api()
    metrics: list[Any] = []

    if api.value_drift_cls is not None:
        metrics.append(api.value_drift_cls(column=feature_name, threshold=threshold))
    elif api.column_drift_metric_cls is not None:
        metrics.append(api.column_drift_metric_cls(column_name=feature_name))
    else:
        raise EvidentlyDriftError(
            "Evidently does not expose a supported column-drift metric in this environment."
        )

    report = _create_report(api, metrics)
    try:
        snapshot = _run_report(report, reference_data, current_data)
        report_dict = _snapshot_to_dict(snapshot, report)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        raise EvidentlyDriftError(
            f"Evidently failed while computing drift for feature '{feature_name}': {exc}"
        ) from exc

    metric_entry = _find_metric_entry(
        report_dict,
        {
            "evidently:metric_v2:ValueDrift",
            "evidently:metric:ColumnDriftMetric",
        },
        feature_name,
    )
    if metric_entry is None:
        raise EvidentlyDriftError(
            f"Evidently report output did not include drift results for column '{feature_name}'."
        )

    metric_value = metric_entry.get("value")
    if isinstance(metric_value, dict):
        p_value = metric_value.get("drift_score", metric_value.get("score"))
        detected = metric_value.get("drift_detected")
    else:
        p_value = metric_value
        detected = None

    p_value_float = float(p_value) if p_value is not None else float("nan")
    detected_bool = bool(detected) if detected is not None else bool(p_value_float < threshold)

    return {
        "feature_name": feature_name,
        "detected": detected_bool,
        "p_value": p_value_float,
        "drift_score": p_value_float,
        "threshold": float(threshold),
        "report": report_dict,
    }


def _build_monitoring_frames(
    config_path: str | Path = "params.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, str | None, list[str]]:
    config = load_config(config_path)
    drift_config = config["drift_monitoring"]
    target = config["raw_data_config"]["target"]

    reference_data_path = resolve_path(drift_config["reference_data_csv"], config_path)
    current_data_path = resolve_path(drift_config["current_data_csv"], config_path)

    predictor = ChurnPredictor(config_path=config_path)

    reference_raw = pd.read_csv(reference_data_path)
    current_raw = pd.read_csv(current_data_path)

    reference_data = reference_raw[predictor.feature_names].copy()
    current_data = current_raw[predictor.feature_names].copy()
    reference_data["prediction"] = predictor.predict_proba(reference_data)
    current_data["prediction"] = predictor.predict_proba(current_data)

    target_column = None
    if target in reference_raw.columns and target in current_raw.columns:
        reference_data[target] = reference_raw[target]
        current_data[target] = current_raw[target]
        target_column = target

    return reference_data, current_data, target_column, predictor.feature_names


def generate_drift_report(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    drift_config = config["drift_monitoring"]
    report_path = resolve_path(drift_config["report_html"], config_path)
    summary_path = resolve_path(drift_config["summary_json"], config_path)

    reference_data, current_data, target_column, feature_names = _build_monitoring_frames(
        config_path=config_path
    )
    api = _load_evidently_api()

    metrics: list[Any] = [api.data_drift_preset_cls()]
    if api.value_drift_cls is not None:
        metrics.append(api.value_drift_cls(column="prediction"))
        if target_column is not None:
            metrics.append(api.value_drift_cls(column=target_column))
    elif api.column_drift_metric_cls is not None:
        metrics.append(api.column_drift_metric_cls(column_name="prediction"))
        if target_column is not None:
            metrics.append(api.column_drift_metric_cls(column_name=target_column))
    else:
        raise EvidentlyDriftError(
            "Evidently does not expose a supported prediction-drift metric in this environment."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    report = _create_report(api, metrics)
    try:
        snapshot = _run_report(report, reference_data, current_data)
        _save_snapshot_html(snapshot, report_path, report)
        report_dict = _snapshot_to_dict(snapshot, report)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        raise EvidentlyDriftError(
            f"Evidently failed while generating the drift report: {exc}"
        ) from exc

    summary = {
        "backend": "evidently",
        "reference_rows": int(len(reference_data)),
        "current_rows": int(len(current_data)),
        "features": feature_names,
        "report_path": str(report_path),
        "prediction_column": "prediction",
        "report": report_dict,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    generate_drift_report(config_path=parsed_args.config)
