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


def validate_dataset(config_path: str | Path = "params.yaml") -> dict:
    config = load_config(config_path)
    data_path = resolve_path(
        config["external_data_config"]["external_data_csv"],
        config_path,
    )
    report_path = resolve_path("reports/data_validation.json", config_path)
    expected_columns = config["raw_data_config"]["model_var"]
    target = config["raw_data_config"]["target"]

    df = pd.read_csv(data_path)
    missing_columns = [column for column in expected_columns if column not in df.columns]
    unexpected_nulls = df[expected_columns].isnull().sum().to_dict()
    null_columns = [column for column, count in unexpected_nulls.items() if count > 0]
    target_classes = sorted(df[target].dropna().unique().tolist())

    report = {
        "data_path": str(data_path),
        "row_count": int(len(df)),
        "expected_columns": expected_columns,
        "missing_columns": missing_columns,
        "null_counts": {key: int(value) for key, value in unexpected_nulls.items()},
        "target_classes": target_classes,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")
    if null_columns:
        raise ValueError(f"Null values found in required columns: {', '.join(null_columns)}")
    if len(target_classes) < 2:
        raise ValueError("Target column must contain at least two classes.")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    validate_dataset(config_path=args.config)
