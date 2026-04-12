from __future__ import annotations

import argparse
import inspect
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.load_data import load_raw_data
from src.data.split_data import split_and_saved_data
from src.evaluation.bootstrap import bootstrap_auc_f1_ci
from src.evaluation.metrics import (
    compute_binary_classification_metrics,
    find_optimal_f1_threshold,
    get_positive_class_scores,
)


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


def get_feat_and_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    x_data = df.drop(columns=[target])
    y_data = df[target]
    return x_data, y_data


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in x_train.columns if column not in numeric_features]

    return ColumnTransformer(
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
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
    )


def encode_binary_target(
    y_data: pd.Series,
    positive_label: str,
) -> tuple[pd.Series, dict[int, str]]:
    unique_labels = pd.Index(pd.Series(y_data).dropna().unique()).tolist()
    negative_candidates = [label for label in unique_labels if label != positive_label]
    if len(negative_candidates) != 1:
        raise ValueError(
            "Extended training expects a binary target with exactly one positive class."
        )

    negative_label = negative_candidates[0]
    encoded_target = y_data.eq(positive_label).astype(int)
    return encoded_target, {0: str(negative_label), 1: str(positive_label)}


def positive_class_ratio(y_train: pd.Series) -> float:
    positive_count = int((pd.Series(y_train) == 1).sum())
    negative_count = int((pd.Series(y_train) == 0).sum())
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


def balanced_class_weights(y_train: pd.Series) -> list[float]:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.asarray(y_train, dtype=int),
    )
    return [float(weight) for weight in weights]


def supports_sample_weight(estimator) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def fit_preprocessed_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    estimator,
    *,
    use_smote: bool,
    use_balanced_sample_weight: bool,
    random_state: int,
) -> Any:
    preprocessor = build_preprocessor(x_train)
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        pipeline = ImbPipeline(steps=steps + [("classifier", estimator)])
    else:
        pipeline = Pipeline(steps=steps + [("classifier", estimator)])

    fit_kwargs: dict[str, Any] = {}
    if use_balanced_sample_weight:
        if not supports_sample_weight(estimator):
            raise ValueError(f"{estimator.__class__.__name__} does not support sample_weight.")
        fit_kwargs["classifier__sample_weight"] = compute_sample_weight(
            class_weight="balanced",
            y=np.asarray(y_train, dtype=int),
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        pipeline.fit(x_train, y_train, **fit_kwargs)

    return pipeline


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 111,
    n_jobs: int = 1,
    use_smote: bool = False,
    use_class_weight: bool = False,
):
    estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced" if use_class_weight else None,
    )
    return fit_preprocessed_model(
        x_train,
        y_train,
        estimator,
        use_smote=use_smote,
        use_balanced_sample_weight=False,
        random_state=random_state,
    )


def train_gradient_boosting(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 111,
    n_jobs: int = 1,
    use_smote: bool = False,
    use_class_weight: bool = False,
):
    del n_jobs
    estimator = GradientBoostingClassifier(random_state=random_state)
    return fit_preprocessed_model(
        x_train,
        y_train,
        estimator,
        use_smote=use_smote,
        use_balanced_sample_weight=use_class_weight,
        random_state=random_state,
    )


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 111,
    n_jobs: int = 1,
    use_smote: bool = False,
    use_class_weight: bool = False,
):
    if XGBClassifier is None:  # pragma: no cover - dependency guard
        raise ImportError("xgboost is not installed.")

    estimator = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        scale_pos_weight=positive_class_ratio(y_train) if use_class_weight else 1.0,
    )
    return fit_preprocessed_model(
        x_train,
        y_train,
        estimator,
        use_smote=use_smote,
        use_balanced_sample_weight=False,
        random_state=random_state,
    )


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 111,
    n_jobs: int = 1,
    use_smote: bool = False,
    use_class_weight: bool = False,
):
    if LGBMClassifier is None:  # pragma: no cover - dependency guard
        raise ImportError("lightgbm is not installed.")

    estimator = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced" if use_class_weight else None,
        verbose=-1,
    )
    return fit_preprocessed_model(
        x_train,
        y_train,
        estimator,
        use_smote=use_smote,
        use_balanced_sample_weight=False,
        random_state=random_state,
    )


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 111,
    n_jobs: int = 1,
    use_smote: bool = False,
    use_class_weight: bool = False,
):
    if CatBoostClassifier is None:  # pragma: no cover - dependency guard
        raise ImportError("catboost is not installed.")

    estimator = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=n_jobs,
        class_weights=balanced_class_weights(y_train) if use_class_weight else None,
    )
    return fit_preprocessed_model(
        x_train,
        y_train,
        estimator,
        use_smote=use_smote,
        use_balanced_sample_weight=False,
        random_state=random_state,
    )


def train_mlp_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 111,
    n_jobs: int = 1,
    use_smote: bool = False,
    use_class_weight: bool = False,
):
    del n_jobs
    estimator = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=400,
        early_stopping=True,
        random_state=random_state,
    )
    return fit_preprocessed_model(
        x_train,
        y_train,
        estimator,
        use_smote=use_smote,
        use_balanced_sample_weight=use_class_weight,
        random_state=random_state,
    )


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    trainer: Callable[..., Any]
    dependency_available: bool = True


def get_model_registry() -> dict[str, ModelSpec]:
    return {
        "random_forest": ModelSpec("random_forest", "Random Forest", train_random_forest),
        "gradient_boosting": ModelSpec(
            "gradient_boosting",
            "Gradient Boosting",
            train_gradient_boosting,
        ),
        "xgboost": ModelSpec(
            "xgboost",
            "XGBoost",
            train_xgboost,
            dependency_available=XGBClassifier is not None,
        ),
        "lightgbm": ModelSpec(
            "lightgbm",
            "LightGBM",
            train_lightgbm,
            dependency_available=LGBMClassifier is not None,
        ),
        "catboost": ModelSpec(
            "catboost",
            "CatBoost",
            train_catboost,
            dependency_available=CatBoostClassifier is not None,
        ),
        "mlp_classifier": ModelSpec(
            "mlp_classifier",
            "MLPClassifier",
            train_mlp_classifier,
        ),
    }


def train_model_for_strategy(
    model_spec: ModelSpec,
    strategy: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    n_jobs: int,
    validation_size: float,
) -> tuple[Any, float, dict[str, Any]]:
    if strategy == "smote":
        model = model_spec.trainer(
            x_train,
            y_train,
            random_state=random_state,
            n_jobs=n_jobs,
            use_smote=True,
            use_class_weight=False,
        )
        return model, 0.5, {"smote": True, "class_weight": False, "threshold_tuned": False}

    if strategy == "class_weight":
        model = model_spec.trainer(
            x_train,
            y_train,
            random_state=random_state,
            n_jobs=n_jobs,
            use_smote=False,
            use_class_weight=True,
        )
        return model, 0.5, {"smote": False, "class_weight": True, "threshold_tuned": False}

    if strategy == "threshold_tuning":
        x_fit, x_valid, y_fit, y_valid = train_test_split(
            x_train,
            y_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train,
        )
        threshold_model = model_spec.trainer(
            x_fit,
            y_fit,
            random_state=random_state,
            n_jobs=n_jobs,
            use_smote=False,
            use_class_weight=False,
        )
        validation_scores = get_positive_class_scores(threshold_model, x_valid, positive_label=1)
        threshold = find_optimal_f1_threshold(y_valid, validation_scores, positive_label=1)
        refit_model = model_spec.trainer(
            x_train,
            y_train,
            random_state=random_state,
            n_jobs=n_jobs,
            use_smote=False,
            use_class_weight=False,
        )
        return refit_model, threshold, {
            "smote": False,
            "class_weight": False,
            "threshold_tuned": True,
        }

    raise ValueError(f"Unsupported strategy: {strategy}")


def evaluate_model_run(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float,
    class_name_map: dict[int, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_scores = get_positive_class_scores(model, x_test, positive_label=1)
    metrics = compute_binary_classification_metrics(
        y_test,
        y_scores,
        threshold=threshold,
        positive_label=1,
        negative_label=0,
        class_name_map=class_name_map,
    )
    cache = {
        "y_true": np.asarray(y_test, dtype=int),
        "y_scores": np.asarray(y_scores, dtype=float),
        "threshold": float(threshold),
    }
    return metrics, cache


def render_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.fillna("n/a").astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_ablation_table(
    comparison_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    *,
    top_k: int = 3,
) -> str:
    completed = comparison_df.loc[comparison_df["status"] == "ok"].copy()
    if completed.empty:
        return "# Model Ablation\n\nNo successful model runs were available.\n"

    top_runs = completed.sort_values(["auc_roc", "pr_auc"], ascending=False).head(top_k)
    bootstrap_columns = [
        "run_id",
        "model",
        "strategy",
        "smote",
        "class_weight",
        "auc_roc_ci_lower",
        "auc_roc_ci_upper",
        "f1_ci_lower",
        "f1_ci_upper",
    ]
    merged = top_runs.merge(
        bootstrap_df[bootstrap_columns] if not bootstrap_df.empty else bootstrap_df,
        on=["run_id", "model", "strategy", "smote", "class_weight"],
        how="left",
    )
    merged["Model"] = merged["model"] + " [" + merged["strategy"] + "]"
    merged["SMOTE"] = merged["smote"].map({True: "Yes", False: "No"})
    merged["class_weight"] = merged["class_weight"].map({True: "Yes", False: "No"})
    merged["AUC-ROC (95% CI)"] = merged.apply(
        lambda row: (
            f"{row['auc_roc']:.4f} "
            f"({row['auc_roc_ci_lower']:.4f}, {row['auc_roc_ci_upper']:.4f})"
        )
        if pd.notna(row["auc_roc_ci_lower"])
        else f"{row['auc_roc']:.4f} (n/a)",
        axis=1,
    )
    merged["F1"] = merged["f1_positive_class"].map(lambda value: f"{value:.4f}")
    merged["PR-AUC"] = merged["pr_auc"].map(lambda value: f"{value:.4f}")

    best_pr_auc_row = completed.sort_values("pr_auc", ascending=False).iloc[0]
    intro = (
        "# Model Ablation\n\n"
        f"Best PR-AUC combination: **{best_pr_auc_row['model']} [{best_pr_auc_row['strategy']}]** "
        f"with PR-AUC **{best_pr_auc_row['pr_auc']:.4f}**.\n\n"
        "Top configurations by AUC-ROC with bootstrap 95% confidence intervals:\n\n"
    )

    ablation_df = merged[
        ["Model", "SMOTE", "class_weight", "AUC-ROC (95% CI)", "F1", "PR-AUC"]
    ].rename(columns={"class_weight": "class_weight"})
    return intro + render_markdown_table(ablation_df) + "\n"


def run_extended_benchmark(
    config_path: str | Path = "params.yaml",
    *,
    output_dir: str | Path = "results",
    model_names: list[str] | None = None,
    strategies: list[str] | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_top_k: int = 3,
    validation_size: float = 0.2,
) -> dict[str, Any]:
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
    results_dir = resolve_path(output_dir, config_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    target = config["raw_data_config"]["target"]
    positive_label = config["raw_data_config"].get("positive_class", "yes")
    random_state = config["raw_data_config"]["random_state"]
    n_jobs = config["training"].get("n_jobs", 1)

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    x_train, y_train_raw = get_feat_and_target(train_df, target)
    x_test, y_test_raw = get_feat_and_target(test_df, target)

    y_train, class_name_map = encode_binary_target(y_train_raw, positive_label)
    y_test, _ = encode_binary_target(y_test_raw, positive_label)

    strategy_list = strategies or ["smote", "class_weight", "threshold_tuning"]
    registry = get_model_registry()
    selected_model_names = model_names or list(registry.keys())

    comparison_rows: list[dict[str, Any]] = []
    evaluation_cache: dict[str, dict[str, Any]] = {}

    for model_key in selected_model_names:
        model_spec = registry[model_key]
        for strategy in strategy_list:
            run_id = f"{model_spec.key}__{strategy}"
            base_row = {
                "run_id": run_id,
                "model_key": model_spec.key,
                "model": model_spec.display_name,
                "strategy": strategy,
                "smote": strategy == "smote",
                "class_weight": strategy == "class_weight",
                "threshold_tuned": strategy == "threshold_tuning",
                "status": "ok",
                "notes": "",
            }

            if not model_spec.dependency_available:
                comparison_rows.append(
                    {
                        **base_row,
                        "status": "skipped",
                        "notes": "Missing optional dependency.",
                    }
                )
                continue

            try:
                model, threshold, strategy_flags = train_model_for_strategy(
                    model_spec,
                    strategy,
                    x_train,
                    y_train,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    validation_size=validation_size,
                )
                metrics, cache = evaluate_model_run(
                    model,
                    x_test,
                    y_test,
                    threshold=threshold,
                    class_name_map=class_name_map,
                )
                evaluation_cache[run_id] = cache
                comparison_rows.append(
                    {
                        **base_row,
                        **strategy_flags,
                        **metrics,
                    }
                )
            except Exception as exc:  # pragma: no cover - execution safety
                comparison_rows.append(
                    {
                        **base_row,
                        "status": "failed",
                        "notes": str(exc),
                    }
                )

    comparison_df = pd.DataFrame(comparison_rows)
    completed_df = comparison_df.loc[comparison_df["status"] == "ok"].copy()
    if not completed_df.empty:
        comparison_df = comparison_df.sort_values(
            by=["status", "pr_auc", "auc_roc"],
            ascending=[True, False, False],
            na_position="last",
        )

    comparison_path = results_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    bootstrap_rows: list[dict[str, Any]] = []
    if not completed_df.empty:
        top_runs = completed_df.sort_values(["auc_roc", "pr_auc"], ascending=False).head(
            bootstrap_top_k
        )
        for _, row in top_runs.iterrows():
            cache = evaluation_cache[row["run_id"]]
            ci = bootstrap_auc_f1_ci(
                cache["y_true"],
                cache["y_scores"],
                threshold=cache["threshold"],
                n_bootstraps=bootstrap_samples,
                random_state=random_state,
            )
            bootstrap_rows.append(
                {
                    "run_id": row["run_id"],
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "smote": bool(row["smote"]),
                    "class_weight": bool(row["class_weight"]),
                    "threshold": float(row["threshold"]),
                    "auc_roc": float(row["auc_roc"]),
                    "f1_positive_class": float(row["f1_positive_class"]),
                    **ci,
                }
            )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_path = results_dir / "bootstrap_ci.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    ablation_markdown = build_ablation_table(
        comparison_df,
        bootstrap_df,
        top_k=bootstrap_top_k,
    )
    ablation_path = results_dir / "ablation_table.md"
    ablation_path.write_text(ablation_markdown, encoding="utf-8")

    best_pr_auc = None
    if not completed_df.empty:
        best_row = completed_df.sort_values("pr_auc", ascending=False).iloc[0]
        best_pr_auc = {
            "model": str(best_row["model"]),
            "strategy": str(best_row["strategy"]),
            "pr_auc": float(best_row["pr_auc"]),
        }

    return {
        "comparison_path": str(comparison_path),
        "bootstrap_path": str(bootstrap_path),
        "ablation_path": str(ablation_path),
        "best_pr_auc": best_pr_auc,
        "comparison_df": comparison_df,
        "bootstrap_df": bootstrap_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-top-k", type=int, default=3)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    run_extended_benchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_top_k=args.bootstrap_top_k,
        validation_size=args.validation_size,
    )
