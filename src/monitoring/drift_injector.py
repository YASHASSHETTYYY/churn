from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftScenario:
    scenario: str
    feature_drifted: str
    description: str


def _get_rng(random_state: int | None = None) -> np.random.Generator:
    return np.random.default_rng(random_state)


def shift_numeric_feature(
    df: pd.DataFrame,
    feature: str,
    shift_amount: float,
    noise_std: float,
    random_state: int | None = None,
) -> pd.DataFrame:
    if feature not in df.columns:
        raise KeyError(f"Unknown feature: {feature}")
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise TypeError(f"Feature '{feature}' must be numeric for numeric shift injection.")

    rng = _get_rng(random_state)
    shifted = df.copy()
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(shifted))
    shifted[feature] = shifted[feature].astype(float) + float(shift_amount) + noise
    return shifted


def shift_categorical_distribution(
    df: pd.DataFrame,
    feature: str,
    target_category: str,
    new_proportion: float,
    random_state: int | None = None,
) -> pd.DataFrame:
    if feature not in df.columns:
        raise KeyError(f"Unknown feature: {feature}")
    if pd.api.types.is_numeric_dtype(df[feature]):
        raise TypeError(f"Feature '{feature}' must be categorical for categorical shift injection.")

    if not 0.0 <= new_proportion <= 1.0:
        raise ValueError("new_proportion must be between 0 and 1.")

    shifted = df.copy()
    rng = _get_rng(random_state)
    categories = shifted[feature].astype(str)
    target_mask = categories == str(target_category)
    desired_target_count = int(round(len(shifted) * new_proportion))
    current_target_count = int(target_mask.sum())

    if desired_target_count == current_target_count:
        return shifted

    if desired_target_count > current_target_count:
        candidates = shifted.index[~target_mask]
        num_to_flip = min(desired_target_count - current_target_count, len(candidates))
        if num_to_flip > 0:
            selected = rng.choice(candidates.to_numpy(), size=num_to_flip, replace=False)
            shifted.loc[selected, feature] = target_category
        return shifted

    non_target_values = categories.loc[~target_mask]
    if non_target_values.empty:
        raise ValueError(
            f"Cannot reduce category '{target_category}' because no alternate categories exist."
        )

    replacement_probs = (
        non_target_values.value_counts(normalize=True)
        .sort_index()
    )
    candidates = shifted.index[target_mask]
    num_to_flip = min(current_target_count - desired_target_count, len(candidates))
    selected = rng.choice(candidates.to_numpy(), size=num_to_flip, replace=False)
    replacements = rng.choice(
        replacement_probs.index.to_numpy(),
        size=num_to_flip,
        replace=True,
        p=replacement_probs.to_numpy(),
    )
    shifted.loc[selected, feature] = replacements
    return shifted


def create_drift_scenario(
    df: pd.DataFrame,
    scenario: str = "gradual",
    magnitude: float = 0.2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, DriftScenario]:
    rng = _get_rng(random_state)
    scenario = scenario.lower()

    if scenario == "gradual":
        feature = "total_day_minutes"
        std = float(df[feature].std(ddof=0) or 1.0)
        progress = np.linspace(0.1, 1.0, len(df))
        shift_profile = magnitude * std * progress
        noise = rng.normal(0.0, 0.03 * std, size=len(df))
        shifted = df.copy()
        shifted[feature] = shifted[feature].astype(float) + shift_profile + noise
        return shifted, DriftScenario(
            scenario="gradual",
            feature_drifted=feature,
            description="Linearly increasing mean shift in daytime usage.",
        )

    if scenario == "sudden":
        feature = "international_plan"
        current_proportion = float((df[feature].astype(str) == "yes").mean())
        new_proportion = min(0.95, max(current_proportion, current_proportion + magnitude))
        shifted = shift_categorical_distribution(
            df,
            feature=feature,
            target_category="yes",
            new_proportion=new_proportion,
            random_state=random_state,
        )
        return shifted, DriftScenario(
            scenario="sudden",
            feature_drifted=feature,
            description="Abrupt increase in the share of international-plan customers.",
        )

    if scenario == "seasonal":
        feature = "number_customer_service_calls"
        std = float(df[feature].std(ddof=0) or 1.0)
        phase = np.linspace(0.0, 2.0 * np.pi, len(df))
        amplitude = max(0.5, magnitude * std * 2.0)
        seasonal_shift = amplitude * (1.0 + np.sin(phase))
        shifted = df.copy()
        shifted[feature] = np.clip(
            np.round(shifted[feature].astype(float) + seasonal_shift),
            a_min=0,
            a_max=None,
        ).astype(int)
        return shifted, DriftScenario(
            scenario="seasonal",
            feature_drifted=feature,
            description="Oscillating support-call volume that mimics seasonal service pressure.",
        )

    raise ValueError(f"Unsupported drift scenario: {scenario}")
