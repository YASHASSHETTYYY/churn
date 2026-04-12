from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.base import clone
from sklearn.metrics import recall_score, roc_auc_score

from src.config import load_config, resolve_path
from src.models.train_extended import (
    encode_binary_target,
    get_feat_and_target,
    get_model_registry,
    train_model_for_strategy,
)


SENSITIVE_FEATURES = ["state", "area_code", "international_plan"]


def safe_auc_roc(y_true: Any, y_score: Any) -> float:
    y_true_array = np.asarray(y_true, dtype=int)
    if np.unique(y_true_array).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true_array, np.asarray(y_score, dtype=float)))


def true_positive_rate(y_true: Any, y_pred: Any) -> float:
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def threshold_scores(y_scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_scores, dtype=float) >= float(threshold)).astype(int)


def load_best_phase1_run(results_path: str | Path) -> pd.Series:
    leaderboard = pd.read_csv(results_path)
    completed = leaderboard.loc[leaderboard["status"] == "ok"].copy()
    if completed.empty:
        raise ValueError("No successful benchmark rows found in results/model_comparison.csv.")
    return completed.sort_values(["pr_auc", "auc_roc"], ascending=False).iloc[0]


def load_training_split(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    train_path = resolve_path(config["processed_data_config"]["train_data_csv"], config_path)
    test_path = resolve_path(config["processed_data_config"]["test_data_csv"], config_path)
    target = config["raw_data_config"]["target"]
    positive_label = config["raw_data_config"].get("positive_class", "yes")
    random_state = int(config["raw_data_config"]["random_state"])
    n_jobs = int(config["training"].get("n_jobs", 1))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    x_train, y_train_raw = get_feat_and_target(train_df, target)
    x_test, y_test_raw = get_feat_and_target(test_df, target)
    y_train, class_name_map = encode_binary_target(y_train_raw, positive_label)
    y_test, _ = encode_binary_target(y_test_raw, positive_label)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "class_name_map": class_name_map,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }


def train_best_phase1_model(
    config_path: str | Path = "params.yaml",
    *,
    validation_size: float = 0.2,
) -> dict[str, Any]:
    best_run = load_best_phase1_run(resolve_path("results/model_comparison.csv", config_path))
    dataset = load_training_split(config_path)
    registry = get_model_registry()
    model_key = str(best_run["model_key"])
    strategy = str(best_run["strategy"])
    model_spec = registry[model_key]
    model, threshold, strategy_flags = train_model_for_strategy(
        model_spec,
        strategy,
        dataset["x_train"],
        dataset["y_train"],
        random_state=dataset["random_state"],
        n_jobs=dataset["n_jobs"],
        validation_size=validation_size,
    )
    return {
        "best_run": best_run,
        "model": model,
        "threshold": float(best_run.get("threshold", threshold)),
        "x_train": dataset["x_train"],
        "y_train": dataset["y_train"],
        "x_test": dataset["x_test"],
        "y_test": dataset["y_test"],
        "strategy_flags": strategy_flags,
    }


def build_metric_frames(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sensitive_series: pd.Series,
) -> tuple[MetricFrame, MetricFrame]:
    binary_frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series,
    )
    auc_frame = MetricFrame(
        metrics={"auc_roc": safe_auc_roc},
        y_true=y_true,
        y_pred=y_score,
        sensitive_features=sensitive_series,
    )
    return binary_frame, auc_frame


def audit_sensitive_feature(
    *,
    feature_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sensitive_series: pd.Series,
) -> pd.DataFrame:
    binary_frame, auc_frame = build_metric_frames(y_true, y_pred, y_score, sensitive_series)
    overall_tpr = float(binary_frame.overall["true_positive_rate"])
    overall_selection_rate = float(binary_frame.overall["selection_rate"])
    overall_fpr = float(binary_frame.overall["false_positive_rate"])
    overall_auc = float(auc_frame.overall["auc_roc"])

    report = binary_frame.by_group.reset_index().rename(
        columns={feature_name: "group"}
    )
    report["auc_roc"] = auc_frame.by_group["auc_roc"].to_numpy()
    counts = sensitive_series.astype(str).value_counts().rename_axis("group").reset_index(name="n_samples")
    report["group"] = report["group"].astype(str)
    report = report.merge(counts, on="group", how="left")
    report["sensitive_feature"] = feature_name
    report["overall_selection_rate"] = overall_selection_rate
    report["overall_true_positive_rate"] = overall_tpr
    report["overall_false_positive_rate"] = overall_fpr
    report["overall_auc_roc"] = overall_auc
    report["tpr_gap_vs_overall"] = (report["true_positive_rate"] - overall_tpr).abs()
    report["potential_disparate_impact"] = report["tpr_gap_vs_overall"] > 0.10
    return report[
        [
            "sensitive_feature",
            "group",
            "n_samples",
            "selection_rate",
            "true_positive_rate",
            "false_positive_rate",
            "auc_roc",
            "overall_selection_rate",
            "overall_true_positive_rate",
            "overall_false_positive_rate",
            "overall_auc_roc",
            "tpr_gap_vs_overall",
            "potential_disparate_impact",
        ]
    ]


def max_tpr_disparity(y_true: pd.Series, y_pred: np.ndarray, sensitive_series: pd.Series) -> float:
    frame = MetricFrame(
        metrics={"true_positive_rate": true_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series,
    )
    overall_tpr = float(frame.overall["true_positive_rate"])
    by_group = frame.by_group["true_positive_rate"]
    return float((by_group - overall_tpr).abs().max())


def choose_mitigation_feature(fairness_report: pd.DataFrame) -> str:
    feature_disparity = (
        fairness_report.groupby("sensitive_feature")["tpr_gap_vs_overall"]
        .max()
        .sort_values(ascending=False)
    )
    return str(feature_disparity.index[0])


def predict_mitigator_scores(mitigator: ExponentiatedGradient, x_data: np.ndarray) -> np.ndarray:
    if hasattr(mitigator, "_pmf_predict"):
        probabilities = np.asarray(mitigator._pmf_predict(x_data), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
    return np.asarray(mitigator.predict(x_data), dtype=float)


def build_tradeoff_markdown(
    *,
    best_run: pd.Series,
    mitigation_feature: str,
    before_auc: float,
    before_disparity: float,
    after_auc: float,
    after_disparity: float,
) -> str:
    auc_delta = after_auc - before_auc
    disparity_delta = after_disparity - before_disparity
    return (
        "# Fairness Mitigation Trade-off\n\n"
        f"Best Phase 1 model: **{best_run['model']} [{best_run['strategy']}]**.\n\n"
        f"Mitigation baseline: **ExponentiatedGradient + DemographicParity** applied to the "
        f"worst proxy group feature, **{mitigation_feature}**.\n\n"
        "| Setting | Overall AUC-ROC | Max TPR disparity |\n"
        "| --- | --- | --- |\n"
        f"| Before mitigation | {before_auc:.4f} | {before_disparity:.4f} |\n"
        f"| After mitigation | {after_auc:.4f} | {after_disparity:.4f} |\n\n"
        "Interpretation:\n"
        f"- AUC-ROC changed by **{auc_delta:+.4f}** after the fairness constraint.\n"
        f"- Maximum TPR disparity changed by **{disparity_delta:+.4f}**.\n"
        "- This baseline is intended as a fairness-aware reference point rather than the final deployment model.\n"
    )


def run_fairness_audit(
    config_path: str | Path = "params.yaml",
    *,
    results_dir: str | Path = "results",
    validation_size: float = 0.2,
) -> dict[str, str]:
    results_root = resolve_path(results_dir, config_path)
    results_root.mkdir(parents=True, exist_ok=True)
    fairness_report_path = results_root / "fairness_report.csv"
    tradeoff_path = results_root / "fairness_tradeoff.md"

    payload = train_best_phase1_model(config_path, validation_size=validation_size)
    model = payload["model"]
    threshold = payload["threshold"]
    x_train = payload["x_train"]
    y_train = payload["y_train"]
    x_test = payload["x_test"]
    y_test = payload["y_test"]
    best_run = payload["best_run"]

    y_score = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    y_pred = threshold_scores(y_score, threshold)

    fairness_frames = []
    for feature_name in SENSITIVE_FEATURES:
        fairness_frames.append(
            audit_sensitive_feature(
                feature_name=feature_name,
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                sensitive_series=x_test[feature_name].astype(str),
            )
        )
    fairness_report = pd.concat(fairness_frames, ignore_index=True)
    fairness_report.to_csv(fairness_report_path, index=False)

    mitigation_feature = choose_mitigation_feature(fairness_report)
    preprocessor = model.named_steps["preprocessor"]
    estimator = clone(model.named_steps["classifier"])
    x_train_transformed = preprocessor.transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    mitigator = ExponentiatedGradient(
        estimator=estimator,
        constraints=DemographicParity(),
    )
    mitigator.fit(
        x_train_transformed,
        np.asarray(y_train, dtype=int),
        sensitive_features=x_train[mitigation_feature].astype(str),
    )

    mitigated_scores = predict_mitigator_scores(mitigator, x_test_transformed)
    mitigated_pred = np.asarray(mitigator.predict(x_test_transformed), dtype=int)

    before_auc = safe_auc_roc(y_test, y_score)
    after_auc = safe_auc_roc(y_test, mitigated_scores)
    before_disparity = max_tpr_disparity(
        y_test,
        y_pred,
        x_test[mitigation_feature].astype(str),
    )
    after_disparity = max_tpr_disparity(
        y_test,
        mitigated_pred,
        x_test[mitigation_feature].astype(str),
    )

    tradeoff_path.write_text(
        build_tradeoff_markdown(
            best_run=best_run,
            mitigation_feature=mitigation_feature,
            before_auc=before_auc,
            before_disparity=before_disparity,
            after_auc=after_auc,
            after_disparity=after_disparity,
        ),
        encoding="utf-8",
    )

    return {
        "fairness_report_path": str(fairness_report_path),
        "tradeoff_path": str(tradeoff_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()
    run_fairness_audit(
        config_path=args.config,
        results_dir=args.results_dir,
        validation_size=args.validation_size,
    )

