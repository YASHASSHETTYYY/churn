from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path


def load_data(data_path, model_var):
    """Load the subset of columns used by the churn model."""

    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    return df[model_var]


def load_raw_data(config_path="params.yaml"):
    """Copy the selected dataset columns into the raw data directory."""

    config = load_config(config_path)
    external_data_path = resolve_path(
        config["external_data_config"]["external_data_csv"],
        config_path,
    )
    raw_data_path = resolve_path(config["raw_data_config"]["raw_data_csv"], config_path)
    model_var = config["raw_data_config"]["model_var"]

    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_data(external_data_path, model_var)
    df.to_csv(raw_data_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)
