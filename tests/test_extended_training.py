from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.models.train_extended import run_extended_benchmark


def test_run_extended_benchmark_writes_expected_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    external_data = pd.read_csv(repo_root / "data" / "external" / "train.csv").head(320)
    external_csv = tmp_path / "external.csv"
    external_data.to_csv(external_csv, index=False)

    config = {
        "external_data_config": {"external_data_csv": str(external_csv)},
        "raw_data_config": {
            "raw_data_csv": str(tmp_path / "raw" / "train.csv"),
            "model_var": external_data.columns.tolist(),
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
            "n_jobs": 1,
        },
    }

    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_extended_benchmark(
        config_path=config_path,
        output_dir="results",
        model_names=["random_forest", "gradient_boosting"],
        bootstrap_samples=25,
        bootstrap_top_k=2,
        validation_size=0.25,
    )

    comparison_path = Path(result["comparison_path"])
    bootstrap_path = Path(result["bootstrap_path"])
    ablation_path = Path(result["ablation_path"])

    assert comparison_path.exists()
    assert bootstrap_path.exists()
    assert ablation_path.exists()

    comparison_df = pd.read_csv(comparison_path)
    bootstrap_df = pd.read_csv(bootstrap_path)
    ablation_markdown = ablation_path.read_text(encoding="utf-8")

    assert {"model", "strategy", "auc_roc", "pr_auc", "confusion_matrix"}.issubset(
        comparison_df.columns
    )
    assert not bootstrap_df.empty
    assert "Model | SMOTE | class_weight | AUC-ROC (95% CI) | F1 | PR-AUC" in ablation_markdown
