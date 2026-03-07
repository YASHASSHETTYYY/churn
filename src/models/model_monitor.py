"""Backward-compatible wrapper around the Evidently drift report job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift_report import generate_drift_report


def model_monitoring(config_path="params.yaml"):
    return generate_drift_report(config_path=config_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_monitoring(config_path=parsed_args.config)
