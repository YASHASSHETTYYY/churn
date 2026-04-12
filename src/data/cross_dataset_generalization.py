from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import resolve_path
from src.evaluation.metrics import (
    compute_binary_classification_metrics,
    get_positive_class_scores,
)
from src.models.train_extended import get_model_registry, train_model_for_strategy


DATASET_SOURCES = {
    "syriatel_telecom": {
        "kind": "local",
        "path": "data/external/train.csv",
    },
    "ibm_telco": {
        "kind": "remote",
        "urls": [
            (
                "https://raw.githubusercontent.com/SaeidRostami/Customer_Churn/master/"
                "WA_Fn-UseC_-Telco-Customer-Churn.csv"
            ),
            (
                "https://huggingface.co/datasets/aai510-group1/customer-churn/raw/main/"
                "WA_Fn-UseC_-Telco-Customer-Churn.csv"
            ),
            (
                "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/"
                "master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
            ),
        ],
        "filename": "ibm_telco_customer_churn.csv",
    },
    "bank_churn": {
        "kind": "remote",
        "urls": [
            "https://raw.githubusercontent.com/nisaamly/Data-Churn/main/Churn_Modelling.csv",
            (
                "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/"
                "master/Churn_Modelling.csv"
            ),
        ],
        "filename": "bank_churn_modelling.csv",
    },
}

COMMON_SCHEMA = {
    "description": "Common schema for cross-dataset churn generalization experiments.",
    "target": "churn",
    "fields": [
        {"name": "dataset_name", "type": "string", "role": "metadata"},
        {"name": "geography_group", "type": "string"},
        {"name": "tenure_months", "type": "float"},
        {"name": "monthly_value", "type": "float"},
        {"name": "total_value", "type": "float"},
        {"name": "support_touches", "type": "float"},
        {"name": "product_count", "type": "float"},
        {"name": "international_plan_flag", "type": "float"},
        {"name": "contract_group", "type": "string"},
        {"name": "churn", "type": "integer", "role": "target"},
    ],
}


def write_schema(schema_path: str | Path) -> Path:
    path = Path(schema_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(COMMON_SCHEMA, indent=2), encoding="utf-8")
    return path


def download_if_needed(dataset_name: str, root_dir: str | Path = "data/external/public") -> Path:
    source = DATASET_SOURCES[dataset_name]
    if source["kind"] == "local":
        return resolve_path(source["path"])

    destination = resolve_path(Path(root_dir) / source["filename"])
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in source["urls"]:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return destination
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        f"Unable to download dataset '{dataset_name}'. Attempts:\n" + "\n".join(errors)
    )


def normalize_yes_no(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(
        {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "y": 1,
            "n": 0,
        }
    ).fillna(0).astype(int)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def harmonize_syriatel(df: pd.DataFrame) -> pd.DataFrame:
    monthly_value = (
        to_numeric(df["total_day_charge"])
        + to_numeric(df["total_eve_charge"])
        + to_numeric(df["total_night_charge"])
        + to_numeric(df["total_intl_charge"])
    )
    tenure = to_numeric(df["account_length"])
    return pd.DataFrame(
        {
            "dataset_name": "syriatel_telecom",
            "geography_group": df["state"].astype(str),
            "tenure_months": tenure,
            "monthly_value": monthly_value,
            "total_value": monthly_value * tenure.clip(lower=1) / 30.0,
            "support_touches": to_numeric(df["number_customer_service_calls"]),
            "product_count": 1.0
            + normalize_yes_no(df["international_plan"])
            + normalize_yes_no(df["voice_mail_plan"]),
            "international_plan_flag": normalize_yes_no(df["international_plan"]),
            "contract_group": df["area_code"].astype(str),
            "churn": normalize_yes_no(df["churn"]),
        }
    )


def harmonize_ibm_telco(df: pd.DataFrame) -> pd.DataFrame:
    total_charges = to_numeric(df["TotalCharges"]).fillna(to_numeric(df["MonthlyCharges"]))
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    product_count = sum(
        normalize_yes_no(
            df[column].replace(
                {
                    "No internet service": "no",
                    "No phone service": "no",
                }
            )
        )
        for column in service_columns
    )
    return pd.DataFrame(
        {
            "dataset_name": "ibm_telco",
            "geography_group": "unknown_us_region",
            "tenure_months": to_numeric(df["tenure"]),
            "monthly_value": to_numeric(df["MonthlyCharges"]),
            "total_value": total_charges,
            "support_touches": normalize_yes_no(
                df["TechSupport"].replace({"No internet service": "no"})
            ).rsub(1),
            "product_count": product_count.astype(float),
            "international_plan_flag": 0.0,
            "contract_group": df["Contract"].astype(str),
            "churn": normalize_yes_no(df["Churn"]),
        }
    )


def harmonize_bank_churn(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_name": "bank_churn",
            "geography_group": df["Geography"].astype(str),
            "tenure_months": to_numeric(df["Tenure"]) * 12.0,
            "monthly_value": to_numeric(df["EstimatedSalary"]) / 12.0,
            "total_value": to_numeric(df["Balance"]),
            "support_touches": 0.0,
            "product_count": to_numeric(df["NumOfProducts"]),
            "international_plan_flag": 0.0,
            "contract_group": (
                df["HasCrCard"]
                .astype(str)
                .map({"1": "credit_card_yes", "0": "credit_card_no"})
                .fillna("credit_card_unknown")
            ),
            "churn": to_numeric(df["Exited"]).fillna(0).astype(int),
        }
    )


def load_harmonized_datasets(
    root_dir: str | Path = "data/external/public",
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_name in DATASET_SOURCES:
        source_path = download_if_needed(dataset_name, root_dir=root_dir)
        frame = pd.read_csv(source_path)
        if dataset_name == "syriatel_telecom":
            frames[dataset_name] = harmonize_syriatel(frame)
        elif dataset_name == "ibm_telco":
            frames[dataset_name] = harmonize_ibm_telco(frame)
        elif dataset_name == "bank_churn":
            frames[dataset_name] = harmonize_bank_churn(frame)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return frames


def get_best_phase1_model() -> tuple[str, str]:
    comparison = pd.read_csv(resolve_path("results/model_comparison.csv"))
    completed = comparison.loc[comparison["status"] == "ok"].copy()
    best = completed.sort_values(["pr_auc", "auc_roc"], ascending=False).iloc[0]
    return str(best["model_key"]), str(best["strategy"])


def run_cross_dataset_generalization(
    *,
    schema_path: str | Path = "data/schema.json",
    output_path: str | Path = "results/cross_dataset_generalization.csv",
    validation_size: float = 0.2,
    random_state: int = 111,
    n_jobs: int = 1,
) -> dict[str, str]:
    schema_file = write_schema(resolve_path(schema_path))
    datasets = load_harmonized_datasets()
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    best_model_key, best_strategy = get_best_phase1_model()
    registry = get_model_registry()
    model_spec = registry[best_model_key]

    rows: list[dict[str, Any]] = []
    feature_columns = [
        field["name"]
        for field in COMMON_SCHEMA["fields"]
        if field["name"] not in {"dataset_name", "churn"}
    ]
    for train_name, train_df in datasets.items():
        x_train = train_df[feature_columns]
        y_train = train_df["churn"].astype(int)
        model, threshold, _ = train_model_for_strategy(
            model_spec,
            best_strategy,
            x_train,
            y_train,
            random_state=random_state,
            n_jobs=n_jobs,
            validation_size=validation_size,
        )

        for test_name, test_df in datasets.items():
            if train_name == test_name:
                continue
            x_test = test_df[feature_columns]
            y_test = test_df["churn"].astype(int)
            y_scores = get_positive_class_scores(model, x_test, positive_label=1)
            metrics = compute_binary_classification_metrics(
                y_test,
                y_scores,
                threshold=threshold,
                positive_label=1,
                negative_label=0,
                class_name_map={0: "no", 1: "yes"},
            )
            rows.append(
                {
                    "model_key": best_model_key,
                    "strategy": best_strategy,
                    "train_dataset": train_name,
                    "test_dataset": test_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "auc_roc": float(metrics["auc_roc"]),
                    "pr_auc": float(metrics["pr_auc"]),
                    "accuracy": float(metrics["accuracy"]),
                    "f1_positive_class": float(metrics["f1_positive_class"]),
                    "threshold": float(threshold),
                }
            )

    report = pd.DataFrame(rows).sort_values(["auc_roc", "pr_auc"], ascending=False)
    report.to_csv(output_file, index=False)
    return {
        "schema_path": str(schema_file),
        "output_path": str(output_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-path", default="data/schema.json")
    parser.add_argument("--output-path", default="results/cross_dataset_generalization.csv")
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()
    run_cross_dataset_generalization(
        schema_path=args.schema_path,
        output_path=args.output_path,
        validation_size=args.validation_size,
    )
