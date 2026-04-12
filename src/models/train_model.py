from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import optuna
except ImportError:  # pragma: no cover - optional in some local envs
    optuna = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.load_data import load_raw_data
from src.data.split_data import split_and_saved_data
from src.evaluation.metrics import compute_binary_classification_metrics, get_positive_class_scores
from src.explainability.generate_shap_artifacts import generate_shap_summary


def get_feat_and_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[target])
    y = df[target]
    return x, y


def ensure_training_data(config_path: str | Path = "params.yaml") -> None:
    config = load_config(config_path)
    raw_data_path = resolve_path(config["raw_data_config"]["raw_data_csv"], config_path)
    train_data_path = resolve_path(
        config["processed_data_config"]["train_data_csv"],
        config_path,
    )
    test_data_path = resolve_path(
        config["processed_data_config"]["test_data_csv"],
        config_path,
    )

    if not raw_data_path.exists():
        load_raw_data(config_path=config_path)
    if not train_data_path.exists() or not test_data_path.exists():
        split_and_saved_data(config_path=config_path)


def get_search_space(trial: optuna.Trial, rf_config: dict) -> dict:
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            rf_config["n_estimators"][0],
            rf_config["n_estimators"][1],
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            rf_config["max_depth"][0],
            rf_config["max_depth"][1],
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            rf_config["min_samples_split"][0],
            rf_config["min_samples_split"][1],
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            rf_config["min_samples_leaf"][0],
            rf_config["min_samples_leaf"][1],
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            rf_config["max_features"],
        ),
    }


def sample_search_space(rng: random.Random, rf_config: dict) -> dict:
    return {
        "n_estimators": rng.randint(
            rf_config["n_estimators"][0],
            rf_config["n_estimators"][1],
        ),
        "max_depth": rng.randint(
            rf_config["max_depth"][0],
            rf_config["max_depth"][1],
        ),
        "min_samples_split": rng.randint(
            rf_config["min_samples_split"][0],
            rf_config["min_samples_split"][1],
        ),
        "min_samples_leaf": rng.randint(
            rf_config["min_samples_leaf"][0],
            rf_config["min_samples_leaf"][1],
        ),
        "max_features": rng.choice(rf_config["max_features"]),
    }


def build_training_pipeline(
    train_x: pd.DataFrame,
    random_state: int,
    n_jobs: int,
    estimator_params: dict,
) -> Pipeline:
    numeric_features = train_x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in train_x.columns if column not in numeric_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
        **estimator_params,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(
    model,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    *,
    positive_label: str,
) -> dict:
    scores = get_positive_class_scores(model, test_x, positive_label=positive_label)
    metrics = compute_binary_classification_metrics(
        test_y,
        scores,
        threshold=0.5,
        positive_label=positive_label,
        negative_label="no" if positive_label == "yes" else 0,
        class_name_map={0: "no", 1: "yes"},
    )
    metrics["f1"] = metrics["f1_positive_class"]
    return metrics


def train_and_evaluate(
    config_path: str | Path = "params.yaml",
    n_trials: int | None = None,
) -> dict:
    ensure_training_data(config_path)
    config = load_config(config_path)
    train_data_path = resolve_path(
        config["processed_data_config"]["train_data_csv"],
        config_path,
    )
    test_data_path = resolve_path(
        config["processed_data_config"]["test_data_csv"],
        config_path,
    )
    artifact_path = resolve_path(config["training"]["artifact_path"], config_path)
    metrics_path = resolve_path(config["training"]["metrics_path"], config_path)
    best_params_path = resolve_path(config["training"]["best_params_path"], config_path)
    shap_plot_path = resolve_path("plots/shap_summary_train.png", config_path)

    target = config["raw_data_config"]["target"]
    positive_label = config["raw_data_config"].get("positive_class", "yes")
    random_state = config["raw_data_config"]["random_state"]
    cv_folds = config["training"]["cv_folds"]
    requested_trials = n_trials or config["training"]["n_trials"]
    n_jobs = config["training"].get("n_jobs", 1)

    tracking_uri = config.get("mlflow", {}).get("tracking_uri")
    experiment_name = config.get("mlflow", {}).get("experiment_name", "churn-training")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="train-random-forest") as run:
        train = pd.read_csv(train_data_path)
        test = pd.read_csv(test_data_path)
        train_x, train_y = get_feat_and_target(train, target)
        test_x, test_y = get_feat_and_target(test, target)

        rf_config = config["random_forest"]
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        best_params: dict | None = None
        best_score = float("-inf")

        def score_params(search_params: dict) -> float:
            candidate_model = build_training_pipeline(
                train_x=train_x,
                random_state=random_state,
                n_jobs=n_jobs,
                estimator_params=search_params,
            )
            scores = cross_val_score(
                candidate_model,
                train_x,
                train_y,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=1,
            )
            return float(scores.mean())

        if optuna is not None:
            def objective(trial: optuna.Trial) -> float:
                search_params = get_search_space(trial, rf_config)
                return score_params(search_params)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            study.optimize(objective, n_trials=requested_trials)
            best_params = study.best_params
            best_score = float(study.best_value)
        else:
            rng = random.Random(random_state)
            for _ in range(requested_trials):
                search_params = sample_search_space(rng, rf_config)
                score = score_params(search_params)
                if score > best_score:
                    best_params = search_params
                    best_score = score

        if best_params is None:
            raise RuntimeError("Unable to determine model hyperparameters.")

        model = build_training_pipeline(
            train_x=train_x,
            random_state=random_state,
            n_jobs=n_jobs,
            estimator_params=best_params,
        )
        model.fit(train_x, train_y)
        metrics = evaluate_model(
            model,
            test_x,
            test_y,
            positive_label=positive_label,
        )
        metrics["best_cv_score"] = best_score
        metrics["n_trials"] = requested_trials

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        best_params_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": model,
            "metadata": {
                "feature_names": train_x.columns.tolist(),
                "feature_dtypes": {
                    column: str(dtype) for column, dtype in train_x.dtypes.items()
                },
                "target": target,
                "positive_label": positive_label,
                "mlflow_run_id": run.info.run_id,
                "model_version": run.info.run_id,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "best_params": best_params,
            "metrics": metrics,
        }
        joblib.dump(bundle, artifact_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

        shap_result = generate_shap_summary(
            model_bundle_path=artifact_path,
            feature_frame=test_x.head(min(len(test_x), 200)),
            output_path=shap_plot_path,
        )

        mlflow.log_params(
            {
                "target": target,
                "positive_label": positive_label,
                "random_state": random_state,
                "cv_folds": cv_folds,
                "n_trials": requested_trials,
                "n_jobs": n_jobs,
                "optimizer": "optuna" if optuna is not None else "random_search",
                **best_params,
            }
        )
        mlflow.log_metrics(
            {
                "auc_roc": float(metrics["auc_roc"]),
                "f1": float(metrics["f1"]),
                "pr_auc": float(metrics["pr_auc"]),
                "accuracy": float(metrics["accuracy"]),
                "best_cv_score": float(best_score),
            }
        )
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")
        mlflow.log_artifact(str(best_params_path), artifact_path="artifacts")
        mlflow.log_artifact(str(metrics_path), artifact_path="artifacts")
        mlflow.log_artifact(str(shap_result["plot_path"]), artifact_path="artifacts/shap")

        return {
            "artifact_path": str(artifact_path),
            "metrics_path": str(metrics_path),
            "best_params_path": str(best_params_path),
            "best_params": best_params,
            "metrics": metrics,
            "mlflow_run_id": run.info.run_id,
            "shap_plot_path": str(shap_result["plot_path"]),
        }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    args.add_argument("--n-trials", type=int, default=None)
    parsed_args = args.parse_args()
    train_and_evaluate(
        config_path=parsed_args.config,
        n_trials=parsed_args.n_trials,
    )
