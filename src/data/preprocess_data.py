from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data.load_data import load_raw_data
from src.data.split_data import split_and_saved_data


def preprocess_data(config_path: str | Path = "params.yaml") -> dict:
    load_raw_data(config_path=config_path)
    split_and_saved_data(config_path=config_path)
    config = load_config(config_path)
    raw_path = resolve_path(config["raw_data_config"]["raw_data_csv"], config_path)
    train_path = resolve_path(config["processed_data_config"]["train_data_csv"], config_path)
    test_path = resolve_path(config["processed_data_config"]["test_data_csv"], config_path)
    summary_path = resolve_path("reports/preprocess_summary.json", config_path)

    raw_df = pd.read_csv(raw_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    summary = {
        "raw_rows": int(len(raw_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "raw_path": str(raw_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    preprocess_data(config_path=args.config)
