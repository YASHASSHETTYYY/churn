from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path


def split_data(df, train_data_path, test_data_path, split_ratio, random_state):
    train, test = train_test_split(
        df,
        test_size=split_ratio,
        random_state=random_state,
        stratify=df.iloc[:, -1],
    )
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")


def split_and_saved_data(config_path="params.yaml"):
    """Split the raw churn dataset and save the train/test partitions."""

    config = load_config(config_path)
    raw_data_path = resolve_path(config["raw_data_config"]["raw_data_csv"], config_path)
    test_data_path = resolve_path(
        config["processed_data_config"]["test_data_csv"],
        config_path,
    )
    train_data_path = resolve_path(
        config["processed_data_config"]["train_data_csv"],
        config_path,
    )
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    random_state = config["raw_data_config"]["random_state"]

    train_data_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(raw_data_path)
    split_data(raw_df, train_data_path, test_data_path, split_ratio, random_state)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)
