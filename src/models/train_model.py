from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
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
) -> dict:
    predictions = model.predict(test_x)
    return {
        "accuracy": accuracy_score(test_y, predictions),
        "precision_weighted": precision_score(
            test_y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            test_y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            test_y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
        "classification_report": classification_report(
            test_y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


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

    target = config["raw_data_config"]["target"]
    positive_label = config["raw_data_config"].get("positive_class", "yes")
    random_state = config["raw_data_config"]["random_state"]
    cv_folds = config["training"]["cv_folds"]
    requested_trials = n_trials or config["training"]["n_trials"]
    n_jobs = config["training"].get("n_jobs", 1)

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
        model = build_training_pipeline(
            train_x=train_x,
            random_state=random_state,
            n_jobs=n_jobs,
            estimator_params=search_params,
        )
        scores = cross_val_score(
            model,
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
    metrics = evaluate_model(model, test_x, test_y)
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
        },
        "best_params": best_params,
        "metrics": metrics,
    }
    joblib.dump(bundle, artifact_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    return {
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "best_params_path": str(best_params_path),
        "best_params": best_params,
        "metrics": metrics,
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
