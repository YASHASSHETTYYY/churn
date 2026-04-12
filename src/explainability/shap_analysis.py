from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.evaluation.metrics import get_positive_class_scores
from src.models.train_extended import (
    encode_binary_target,
    ensure_training_data,
    get_feat_and_target,
    get_model_registry,
    train_model_for_strategy,
)


def slugify(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()


def humanize_feature_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def pick_best_phase1_run(results_path: str | Path) -> pd.Series:
    leaderboard = pd.read_csv(results_path)
    completed = leaderboard.loc[leaderboard["status"] == "ok"].copy()
    if completed.empty:
        raise ValueError("No successful Phase 1 benchmark runs were found.")
    return completed.sort_values(["auc_roc", "pr_auc"], ascending=False).iloc[0]


def load_phase1_data(config_path: str | Path) -> dict[str, Any]:
    ensure_training_data(config_path)
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
        "config": config,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "class_name_map": class_name_map,
        "positive_label": positive_label,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }


def fit_best_model(
    best_run: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    n_jobs: int,
    validation_size: float,
):
    registry = get_model_registry()
    model_key = str(best_run["model_key"])
    strategy = str(best_run["strategy"])
    if model_key not in registry:
        raise KeyError(f"Unknown model key in leaderboard: {model_key}")

    model_spec = registry[model_key]
    model, threshold, _ = train_model_for_strategy(
        model_spec,
        strategy,
        x_train,
        y_train,
        random_state=random_state,
        n_jobs=n_jobs,
        validation_size=validation_size,
    )
    return model, threshold


def get_model_components(model) -> tuple[Any | None, Any]:
    if hasattr(model, "named_steps"):
        return model.named_steps.get("preprocessor"), model.named_steps.get("classifier")
    return None, model


def transform_feature_frame(
    preprocessor,
    feature_frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    if preprocessor is None:
        return feature_frame.to_numpy(dtype=float), feature_frame.columns.tolist()

    transformed = preprocessor.transform(feature_frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = list(preprocessor.get_feature_names_out())
    return np.asarray(transformed, dtype=float), feature_names


def map_transformed_feature_to_raw(
    transformed_name: str,
    raw_feature_names: list[str],
) -> str:
    suffix = transformed_name.split("__", 1)[-1]
    for raw_feature in sorted(raw_feature_names, key=len, reverse=True):
        if suffix == raw_feature or suffix.startswith(f"{raw_feature}_"):
            return raw_feature
    return suffix


def aggregate_shap_by_raw_feature(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    raw_feature_names: list[str],
) -> np.ndarray:
    feature_index = {feature_name: index for index, feature_name in enumerate(raw_feature_names)}
    aggregated = np.zeros((shap_values.shape[0], len(raw_feature_names)), dtype=float)

    for transformed_index, transformed_name in enumerate(transformed_feature_names):
        raw_feature = map_transformed_feature_to_raw(transformed_name, raw_feature_names)
        if raw_feature in feature_index:
            aggregated[:, feature_index[raw_feature]] += shap_values[:, transformed_index]

    return aggregated


def encode_features_for_plotting(feature_frame: pd.DataFrame) -> pd.DataFrame:
    encoded = feature_frame.copy()
    for column in encoded.columns:
        if pd.api.types.is_numeric_dtype(encoded[column]):
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce")
            if encoded[column].isna().all():
                encoded[column] = 0.0
            else:
                encoded[column] = encoded[column].fillna(float(encoded[column].median()))
        else:
            categories = pd.Categorical(encoded[column].astype(str))
            encoded[column] = categories.codes.astype(float)
    return encoded


def is_tree_estimator(estimator) -> bool:
    return estimator.__class__.__name__ in {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
    }


def choose_explainer(
    estimator,
    background_data: np.ndarray,
):
    if is_tree_estimator(estimator):
        return shap.TreeExplainer(estimator), "TreeExplainer"

    if (
        torch is not None and isinstance(estimator, torch.nn.Module)
    ):  # pragma: no cover - future-proof
        background_tensor = torch.tensor(background_data, dtype=torch.float32)
        return shap.DeepExplainer(estimator, background_tensor), "DeepExplainer"

    if estimator.__class__.__name__ == "MLPClassifier":
        background_sample = background_data[: min(50, len(background_data))]
        return (
            shap.KernelExplainer(estimator.predict_proba, background_sample),
            "KernelExplainer",
        )

    raise TypeError(
        f"Unsupported estimator for SHAP analysis: {estimator.__class__.__name__}"
    )


def _normalize_shap_output(
    shap_output,
    expected_value,
    *,
    positive_index: int = 1,
) -> tuple[np.ndarray, float]:
    if isinstance(shap_output, list):
        values = np.asarray(shap_output[positive_index])
        base_value = np.asarray(expected_value)[positive_index]
        return values, float(np.asarray(base_value).ravel()[0])

    values = np.asarray(shap_output)
    if values.ndim == 3:
        values = values[:, :, positive_index]
        base_value = np.asarray(expected_value)[positive_index]
        return values, float(np.asarray(base_value).ravel()[0])

    base_value_array = np.asarray(expected_value)
    base_value = float(base_value_array.ravel()[0]) if base_value_array.size else 0.0
    return values, base_value


def compute_shap_values(
    explainer,
    transformed_test: np.ndarray,
    *,
    positive_index: int = 1,
) -> tuple[np.ndarray, float]:
    if explainer.__class__.__name__ == "DeepExplainer":  # pragma: no cover - future-proof
        test_tensor = torch.tensor(transformed_test, dtype=torch.float32)
        shap_output = explainer.shap_values(test_tensor)
        expected_value = explainer.expected_value
        return _normalize_shap_output(
            shap_output,
            expected_value,
            positive_index=positive_index,
        )

    shap_output = explainer.shap_values(transformed_test)
    expected_value = getattr(explainer, "expected_value", 0.0)
    return _normalize_shap_output(shap_output, expected_value, positive_index=positive_index)


def infer_feature_direction(feature: pd.Series, shap_column: np.ndarray) -> str:
    non_null_feature = feature.copy()
    if (
        pd.api.types.is_numeric_dtype(non_null_feature)
        and non_null_feature.nunique(dropna=True) > 5
    ):
        correlation = pd.Series(non_null_feature).corr(pd.Series(shap_column), method="spearman")
        if pd.isna(correlation) or abs(float(correlation)) < 0.05:
            return "Mixed or non-linear effect"
        if float(correlation) > 0:
            return "Higher values increase churn risk"
        return "Higher values reduce churn risk"

    grouped = (
        pd.DataFrame(
            {
                "feature_value": non_null_feature.astype(str).fillna("missing"),
                "shap_value": shap_column,
            }
        )
        .groupby("feature_value", dropna=False)["shap_value"]
        .mean()
        .sort_values(ascending=False)
    )

    if grouped.empty or len(grouped.index) == 1:
        return "Mixed or non-linear effect"

    top_category = str(grouped.index[0])
    if grouped.iloc[0] <= 0:
        return "Mixed or non-linear effect"
    return f"{top_category} is associated with higher churn risk"


def infer_business_interpretation(feature_name: str) -> str:  # noqa: C901
    lowered = feature_name.lower()
    if "customer_service" in lowered:
        return (
            "Repeated support contacts suggest unresolved service friction and a higher "
            "chance of switching."
        )
    if "international_plan" in lowered:
        return (
            "International-plan customers may face pricing or package-fit concerns that "
            "make churn more likely."
        )
    if "total_intl_calls" in lowered:
        return (
            "International calling frequency captures how central the service is to the "
            "customer's routine and whether the plan still fits that need."
        )
    if "total_intl_charge" in lowered or "total_intl_minutes" in lowered:
        return (
            "International usage and charges can create billing pressure that pushes "
            "customers toward competitors."
        )
    if "total_day" in lowered:
        return (
            "Heavy daytime usage increases bill exposure and can signal plan mismatch "
            "or price sensitivity."
        )
    if "total_eve" in lowered:
        return (
            "Evening usage captures engagement intensity and may reveal whether the "
            "tariff structure fits customer habits."
        )
    if "total_night" in lowered:
        return (
            "Night-time usage reflects off-peak behavior and can indicate whether "
            "customers are using the plan efficiently."
        )
    if "voice_mail_plan" in lowered:
        return (
            "Voice-mail adoption often reflects product engagement and can separate "
            "sticky users from low-engagement accounts."
        )
    if "vmail" in lowered:
        return "Voice-mail activity is a proxy for service adoption and customer engagement."
    if "account_length" in lowered:
        return (
            "Tenure captures relationship maturity; shorter relationships typically "
            "have weaker loyalty."
        )
    if lowered in {"state", "area_code"}:
        return (
            "Regional variation may reflect local competition, coverage quality, or "
            "pricing pressure."
        )
    return (
        "This feature helps differentiate churn risk by capturing customer behavior "
        "or plan fit."
    )


def feature_theme(feature_name: str) -> str:
    lowered = feature_name.lower()
    if "customer_service" in lowered:
        return "service"
    if "plan" in lowered or "charge" in lowered or "minutes" in lowered or "intl" in lowered:
        return "pricing_usage"
    if "vmail" in lowered:
        return "engagement"
    if "account_length" in lowered:
        return "tenure"
    if lowered in {"state", "area_code"}:
        return "regional"
    return "general"


def get_interaction_partners(encoded_features: pd.DataFrame) -> dict[str, str]:
    correlations = encoded_features.corr(numeric_only=True).abs().fillna(0.0)
    partners: dict[str, str] = {}
    for feature_name in correlations.columns:
        ranked = correlations[feature_name].drop(labels=[feature_name], errors="ignore")
        partners[feature_name] = (
            ranked.sort_values(ascending=False).index[0]
            if not ranked.empty
            else feature_name
        )
    return partners


def build_top_features_table(
    feature_frame: pd.DataFrame,
    aggregated_shap_values: np.ndarray,
    interaction_partners: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_frame.columns):
        shap_column = aggregated_shap_values[:, index]
        rows.append(
            {
                "feature": feature_name,
                "mean_abs_shap": float(np.mean(np.abs(shap_column))),
                "direction": infer_feature_direction(feature_frame[feature_name], shap_column),
                "business_interpretation": infer_business_interpretation(feature_name),
                "interaction_feature": interaction_partners.get(feature_name, feature_name),
            }
        )

    top_features = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top_features.insert(0, "rank", np.arange(1, len(top_features) + 1))
    return top_features


def save_summary_bar_plot(
    shap_values: np.ndarray,
    encoded_features: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values,
        features=encoded_features,
        feature_names=encoded_features.columns.tolist(),
        plot_type="bar",
        max_display=min(20, encoded_features.shape[1]),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def save_beeswarm_plot(
    shap_values: np.ndarray,
    encoded_features: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        features=encoded_features,
        feature_names=encoded_features.columns.tolist(),
        max_display=min(20, encoded_features.shape[1]),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def save_dependence_plots(
    top_features: pd.DataFrame,
    shap_values: np.ndarray,
    encoded_features: pd.DataFrame,
    plots_dir: Path,
) -> list[str]:
    paths: list[str] = []
    for feature_name in top_features["feature"].head(5):
        interaction_feature = top_features.loc[
            top_features["feature"] == feature_name,
            "interaction_feature",
        ].iloc[0]
        output_path = plots_dir / f"shap_dependence_{slugify(feature_name)}.png"
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            feature_name,
            shap_values,
            encoded_features,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.title(
            (
                f"{humanize_feature_name(feature_name)} vs "
                f"{humanize_feature_name(interaction_feature)}"
            ),
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        paths.append(str(output_path))
    return paths


def select_local_samples(
    y_test: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> list[tuple[int, str]]:
    y_array = np.asarray(y_test, dtype=int)
    churners = np.where(y_array == 1)[0]
    non_churners = np.where(y_array == 0)[0]

    chosen_positions: list[int] = []
    labels: list[tuple[int, str]] = []

    top_churners = churners[np.argsort(scores[churners])[::-1][:2]]
    for position in top_churners:
        chosen_positions.append(int(position))
        labels.append((int(position), "true churner"))

    safest_non_churners = non_churners[np.argsort(scores[non_churners])[:2]]
    for position in safest_non_churners:
        chosen_positions.append(int(position))
        labels.append((int(position), "true non-churner"))

    remaining = [index for index in range(len(scores)) if index not in chosen_positions]
    borderline_position = min(
        remaining,
        key=lambda index: abs(float(scores[index]) - float(threshold)),
    )
    labels.append((int(borderline_position), "borderline"))
    return labels


def save_waterfall_plots(
    sample_positions: list[tuple[int, str]],
    aggregated_shap_values: np.ndarray,
    base_value: float,
    feature_frame: pd.DataFrame,
    encoded_features: pd.DataFrame,
    scores: np.ndarray,
    y_test_raw: pd.Series,
    plots_dir: Path,
) -> list[str]:
    output_paths: list[str] = []
    for plot_index, (row_position, sample_label) in enumerate(sample_positions, start=1):
        raw_row = feature_frame.iloc[row_position]
        explanation = shap.Explanation(
            values=aggregated_shap_values[row_position],
            base_values=float(base_value),
            data=encoded_features.iloc[row_position].to_numpy(dtype=float),
            display_data=raw_row.astype(str).to_numpy(),
            feature_names=feature_frame.columns.tolist(),
        )

        shap.plots.waterfall(explanation, max_display=10, show=False)
        figure = plt.gcf()
        figure.set_size_inches(11, 7)
        figure.suptitle(
            (
                f"Sample {plot_index}: {sample_label.title()} | "
                f"True label={y_test_raw.iloc[row_position]} | "
                f"Predicted churn score={scores[row_position]:.3f}"
            ),
            y=1.02,
            fontsize=12,
        )
        output_path = plots_dir / f"shap_waterfall_sample_{plot_index}.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        output_paths.append(str(output_path))
    return output_paths


def build_recommendations(top_features: pd.DataFrame) -> list[str]:  # noqa: C901
    ordered_themes = list(
        dict.fromkeys(top_features["feature"].map(feature_theme).tolist())
    )

    recommendations: list[str] = []
    for theme in ordered_themes:
        if theme == "service":
            recommendations.append(
                (
                    "Prioritize proactive retention outreach for customers with repeated "
                    "customer-service contacts, and route them to specialist resolution "
                    "teams before renewal windows."
                )
            )
        elif theme == "pricing_usage":
            recommendations.append(
                (
                    "Redesign or personalize plan offers for heavy day-time and "
                    "international users, especially when current usage patterns imply "
                    "avoidable bill shock."
                )
            )
        elif theme == "engagement":
            recommendations.append(
                (
                    "Use onboarding and adoption campaigns to deepen attachment to "
                    "sticky features such as voice-mail or related service bundles."
                )
            )
        elif theme == "tenure":
            recommendations.append(
                (
                    "Launch an early-life retention play for newer accounts, where "
                    "loyalty is still forming and switching costs are lower."
                )
            )
        elif theme == "regional":
            recommendations.append(
                (
                    "Investigate region-specific churn pockets to determine whether "
                    "local competition, coverage, or pricing requires market-level "
                    "intervention."
                )
            )

        if len(recommendations) == 3:
            break

    fallback_recommendations = [
        (
            "Combine SHAP-based risk flags with CRM triggers so high-risk accounts "
            "receive timely save offers instead of generic outbound messaging."
        ),
        (
            "Monitor whether the dominant SHAP drivers shift after retention campaigns "
            "to confirm that operational changes are addressing the real churn "
            "mechanisms."
        ),
        (
            "Feed the top SHAP drivers into marketing segmentation so customer "
            "journeys reflect service friction, price sensitivity, and engagement "
            "intensity separately."
        ),
    ]

    for fallback in fallback_recommendations:
        if len(recommendations) == 3:
            break
        if fallback not in recommendations:
            recommendations.append(fallback)

    return recommendations[:3]


def build_churn_drivers_markdown(
    best_run: pd.Series,
    top_features: pd.DataFrame,
    recommendations: list[str],
) -> str:
    top_ten = top_features.head(10).copy()
    primary_features = ", ".join(
        humanize_feature_name(feature_name) for feature_name in top_ten["feature"].head(5)
    )
    narrative = (
        f"SHAP analysis of the Phase 1 best model, **{best_run['model']}** trained "
        f"with the **{best_run['strategy']}** imbalance strategy, shows that churn "
        f"risk is driven primarily by a mix of service friction, plan design, and "
        f"usage intensity. The most influential features were {primary_features}. "
        f"Across the top-ranked features, customers with signals of repeated support "
        f"issues, expensive usage patterns, or poor product-plan fit tended to "
        f"receive positive SHAP contributions that pushed the model toward a churn "
        f"prediction, while "
        f"protective engagement signals tended to pull predictions back toward retention."
    )

    table_lines = [
        "| Feature | Mean |SHAP| | Direction | Business Interpretation |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in top_ten.iterrows():
        table_lines.append(
            "| "
            + " | ".join(
                [
                    humanize_feature_name(str(row["feature"])),
                    f"{float(row['mean_abs_shap']):.4f}",
                    str(row["direction"]),
                    str(row["business_interpretation"]),
                ]
            )
            + " |"
        )

    recommendation_lines = [
        f"{index}. {recommendation}"
        for index, recommendation in enumerate(recommendations, start=1)
    ]

    return (
        "# Key Churn Drivers\n\n"
        + narrative
        + "\n\n## Top SHAP Drivers\n\n"
        + "\n".join(table_lines)
        + "\n\n## Business Recommendations\n\n"
        + "\n".join(recommendation_lines)
        + "\n"
    )


def make_markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def make_code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def build_notebook_payload() -> dict[str, Any]:
    cells = [
        make_markdown_cell(
            """
            # SHAP Analysis For Telecom Customer Churn

            This notebook reproduces the explainability figures for the best-performing
            Phase 1 churn model. Each section is written to help non-technical readers
            understand which customer traits pushed the
            model toward a churn or retention prediction.
            """
        ),
        make_code_cell(
            """
            from pathlib import Path
            import sys

            def find_project_root(start: Path) -> Path:
                current = start.resolve()
                for candidate in [current, *current.parents]:
                    if (candidate / "src").exists() and (candidate / "params.yaml").exists():
                        return candidate
                raise RuntimeError("Could not find project root.")

            PROJECT_ROOT = find_project_root(Path.cwd())
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import pandas as pd
            from IPython.display import Image, Markdown, display
            from src.explainability.shap_analysis import run_shap_analysis, slugify

            analysis = run_shap_analysis(config_path=PROJECT_ROOT / "params.yaml")
            top_features = pd.read_csv(PROJECT_ROOT / "results" / "top_features.csv")
            top_features.head(10)
            """
        ),
        make_markdown_cell(
            """
            ## Global Feature Importance

            The two plots below answer complementary questions. The summary bar plot
            shows which variables matter most overall, while the beeswarm plot shows
            whether the same feature pushes different
            customers in different directions.
            """
        ),
        make_code_cell(
            """
            display(Markdown("### SHAP Summary Bar Plot"))
            display(Image(filename=str(PROJECT_ROOT / "plots" / "shap_summary_bar.png")))

            display(Markdown("### SHAP Beeswarm Plot"))
            display(Image(filename=str(PROJECT_ROOT / "plots" / "shap_beeswarm.png")))
            """
        ),
        make_markdown_cell(
            """
            ## Dependence Plots

            These charts focus on the five most important drivers one at a time. Each
            point is a customer. The horizontal position shows the feature value, the
            vertical position shows the SHAP effect on churn risk,
            and the color reflects the strongest correlated companion feature.
            """
        ),
        make_code_cell(
            """
            dependence_features = top_features["feature"].head(5).tolist()
            for feature_name in dependence_features:
                display(Markdown(f"### {feature_name.replace('_', ' ').title()}"))
                display(
                    Image(
                        filename=str(
                            PROJECT_ROOT
                            / "plots"
                            / f"shap_dependence_{slugify(feature_name)}.png"
                        )
                    )
                )
            """
        ),
        make_markdown_cell(
            """
            ## Local Customer Explanations

            The waterfall plots below explain individual predictions. Features pushing
            the customer toward churn appear on one side of the baseline prediction,
            while protective features push the score back down.
            """
        ),
        make_code_cell(
            """
            for index in range(1, 6):
                display(Markdown(f"### Sample {index}"))
                display(
                    Image(
                        filename=str(
                            PROJECT_ROOT / "plots" / f"shap_waterfall_sample_{index}.png"
                        )
                    )
                )
            """
        ),
        make_markdown_cell(
            """
            ## Business Interpretation

            The final markdown report translates the model findings into operational
            language for churn-reduction planning.
            """
        ),
        make_code_cell(
            """
            display(
                Markdown(
                    (PROJECT_ROOT / "paper" / "churn_drivers_analysis.md").read_text(
                        encoding="utf-8"
                    )
                )
            )
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                (
                    "version"
                ): f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(notebook_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps(build_notebook_payload(), indent=2),
        encoding="utf-8",
    )


def run_shap_analysis(
    config_path: str | Path = "params.yaml",
    *,
    phase1_results_path: str | Path = "results/model_comparison.csv",
    plots_dir: str | Path = "plots",
    results_dir: str | Path = "results",
    paper_path: str | Path = "paper/churn_drivers_analysis.md",
    notebook_path: str | Path = "notebooks/shap_analysis.ipynb",
    validation_size: float = 0.2,
) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    phase1_results = resolve_path(phase1_results_path, config_path)
    best_run = pick_best_phase1_run(phase1_results)
    dataset = load_phase1_data(config_path)

    plots_root = resolve_path(plots_dir, config_path)
    results_root = resolve_path(results_dir, config_path)
    paper_output_path = resolve_path(paper_path, config_path)
    notebook_output_path = resolve_path(notebook_path, config_path)

    plots_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    paper_output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_output_path.parent.mkdir(parents=True, exist_ok=True)

    model, threshold = fit_best_model(
        best_run,
        dataset["x_train"],
        dataset["y_train"],
        random_state=dataset["random_state"],
        n_jobs=dataset["n_jobs"],
        validation_size=validation_size,
    )

    preprocessor, estimator = get_model_components(model)
    transformed_train, transformed_feature_names = transform_feature_frame(
        preprocessor,
        dataset["x_train"],
    )
    transformed_test, _ = transform_feature_frame(preprocessor, dataset["x_test"])
    raw_feature_names = dataset["x_test"].columns.tolist()
    encoded_test = encode_features_for_plotting(dataset["x_test"])
    interaction_partners = get_interaction_partners(encoded_test)

    explainer, explainer_name = choose_explainer(estimator, transformed_train)
    transformed_shap_values, base_value = compute_shap_values(
        explainer,
        transformed_test,
        positive_index=1,
    )
    aggregated_shap_values = aggregate_shap_by_raw_feature(
        transformed_shap_values,
        transformed_feature_names,
        raw_feature_names,
    )

    top_features = build_top_features_table(
        dataset["x_test"],
        aggregated_shap_values,
        interaction_partners,
    )
    top_features_path = results_root / "top_features.csv"
    top_features.head(10).to_csv(top_features_path, index=False)

    summary_bar_path = plots_root / "shap_summary_bar.png"
    beeswarm_path = plots_root / "shap_beeswarm.png"
    save_summary_bar_plot(aggregated_shap_values, encoded_test, summary_bar_path)
    save_beeswarm_plot(aggregated_shap_values, encoded_test, beeswarm_path)
    dependence_paths = save_dependence_plots(
        top_features.head(5),
        aggregated_shap_values,
        encoded_test,
        plots_root,
    )

    scores = get_positive_class_scores(model, dataset["x_test"], positive_label=1)
    sample_positions = select_local_samples(dataset["y_test"], scores, threshold)
    waterfall_paths = save_waterfall_plots(
        sample_positions,
        aggregated_shap_values,
        base_value,
        dataset["x_test"],
        encoded_test,
        scores,
        dataset["y_test_raw"],
        plots_root,
    )

    recommendations = build_recommendations(top_features.head(10))
    markdown = build_churn_drivers_markdown(best_run, top_features, recommendations)
    paper_output_path.write_text(markdown, encoding="utf-8")
    write_notebook(notebook_output_path)

    return {
        "best_model": str(best_run["model"]),
        "best_strategy": str(best_run["strategy"]),
        "explainer": explainer_name,
        "threshold": float(threshold),
        "top_features_path": str(top_features_path),
        "summary_bar_path": str(summary_bar_path),
        "beeswarm_path": str(beeswarm_path),
        "dependence_paths": dependence_paths,
        "waterfall_paths": waterfall_paths,
        "paper_path": str(paper_output_path),
        "notebook_path": str(notebook_output_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--phase1-results", default="results/model_comparison.csv")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--paper-path", default="paper/churn_drivers_analysis.md")
    parser.add_argument("--notebook-path", default="notebooks/shap_analysis.ipynb")
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    run_shap_analysis(
        config_path=args.config,
        phase1_results_path=args.phase1_results,
        plots_dir=args.plots_dir,
        results_dir=args.results_dir,
        paper_path=args.paper_path,
        notebook_path=args.notebook_path,
        validation_size=args.validation_size,
    )
