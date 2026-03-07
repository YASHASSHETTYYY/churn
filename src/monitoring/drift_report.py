from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.models.predict import ChurnPredictor


def _save_fallback_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path: Path,
    summary_path: Path,
    error_message: str,
) -> dict:
    rows = []
    for column in reference_data.columns:
        reference_series = reference_data[column]
        current_series = current_data[column]
        if pd.api.types.is_numeric_dtype(reference_series):
            reference_mean = float(reference_series.mean())
            current_mean = float(current_series.mean())
            rows.append(
                {
                    "column": column,
                    "reference_mean": reference_mean,
                    "current_mean": current_mean,
                    "mean_delta": current_mean - reference_mean,
                }
            )
        else:
            reference_mode = reference_series.mode().iloc[0]
            current_mode = current_series.mode().iloc[0]
            rows.append(
                {
                    "column": column,
                    "reference_mode": str(reference_mode),
                    "current_mode": str(current_mode),
                }
            )

    summary_frame = pd.DataFrame(rows)
    html = (
        "<html><body>"
        "<h1>Customer Churn Drift Report</h1>"
        "<p>Evidently could not run in this environment. "
        f"Fallback summary generated instead.</p><pre>{error_message}</pre>"
        f"{summary_frame.to_html(index=False)}"
        "</body></html>"
    )
    report_path.write_text(html, encoding="utf-8")

    summary = {
        "backend": "fallback",
        "report_path": str(report_path),
        "error": error_message,
        "columns": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate_drift_report(config_path: str | Path = "params.yaml") -> dict:
    config = load_config(config_path)
    drift_config = config["drift_monitoring"]
    target = config["raw_data_config"]["target"]

    reference_data_path = resolve_path(
        drift_config["reference_data_csv"],
        config_path,
    )
    current_data_path = resolve_path(
        drift_config["current_data_csv"],
        config_path,
    )
    report_path = resolve_path(drift_config["report_html"], config_path)
    summary_path = resolve_path(drift_config["summary_json"], config_path)

    predictor = ChurnPredictor(config_path=config_path)

    reference_raw = pd.read_csv(reference_data_path)
    current_raw = pd.read_csv(current_data_path)

    reference_data = reference_raw[predictor.feature_names].copy()
    current_data = current_raw[predictor.feature_names].copy()
    reference_data["prediction"] = predictor.predict_proba(reference_data)
    current_data["prediction"] = predictor.predict_proba(current_data)

    if target in reference_raw.columns and target in current_raw.columns:
        reference_data[target] = reference_raw[target]
        current_data[target] = current_raw[target]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from evidently.metrics import ColumnDriftMetric
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report

        metrics = [DataDriftPreset(), ColumnDriftMetric(column_name="prediction")]
        if target in reference_raw.columns and target in current_raw.columns:
            metrics.append(TargetDriftPreset())

        report = Report(metrics=metrics)
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html(str(report_path))

        summary = {
            "backend": "evidently",
            "reference_rows": int(len(reference_data)),
            "current_rows": int(len(current_data)),
            "features": predictor.feature_names,
            "report_path": str(report_path),
            "prediction_column": "prediction",
            "report": report.as_dict(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    except Exception as exc:
        return _save_fallback_report(
            reference_data=reference_data,
            current_data=current_data,
            report_path=report_path,
            summary_path=summary_path,
            error_message=str(exc),
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    generate_drift_report(config_path=parsed_args.config)
