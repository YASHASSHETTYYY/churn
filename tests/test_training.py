from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.models.train_model import train_and_evaluate


def test_train_and_evaluate_writes_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    external_data = pd.read_csv(repo_root / "data" / "external" / "train.csv").head(240)
    external_csv = tmp_path / "external.csv"
    external_data.to_csv(external_csv, index=False)

    config = {
        "external_data_config": {"external_data_csv": str(external_csv)},
        "raw_data_config": {
            "raw_data_csv": str(tmp_path / "raw" / "train.csv"),
            "model_var": [
                "churn",
                "number_vmail_messages",
                "total_day_calls",
                "total_eve_minutes",
                "total_eve_charge",
                "total_intl_minutes",
                "number_customer_service_calls",
            ],
            "train_test_split_ratio": 0.2,
            "target": "churn",
            "positive_class": "yes",
            "random_state": 111,
        },
        "processed_data_config": {
            "train_data_csv": str(tmp_path / "processed" / "train.csv"),
            "test_data_csv": str(tmp_path / "processed" / "test.csv"),
        },
        "training": {
            "artifact_path": str(tmp_path / "models" / "churn_model.joblib"),
            "metrics_path": str(tmp_path / "reports" / "metrics.json"),
            "best_params_path": str(tmp_path / "reports" / "best_params.json"),
            "n_trials": 2,
            "cv_folds": 3,
        },
        "random_forest": {
            "n_estimators": [10, 20],
            "max_depth": [2, 6],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        },
        "model_dir": str(tmp_path / "webapp" / "model.joblib"),
        "drift_monitoring": {
            "reference_data_csv": str(tmp_path / "processed" / "train.csv"),
            "current_data_csv": str(tmp_path / "processed" / "test.csv"),
            "report_html": str(tmp_path / "reports" / "drift_report.html"),
            "summary_json": str(tmp_path / "reports" / "drift_report.json"),
        },
    }

    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_and_evaluate(config_path=config_path, n_trials=2)

    assert Path(result["artifact_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["best_params_path"]).exists()
    assert "f1_weighted" in result["metrics"]
