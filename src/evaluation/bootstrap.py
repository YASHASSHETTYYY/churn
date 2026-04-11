from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def bootstrap_auc_f1_ci(
    y_true,
    y_scores,
    *,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    random_state: int = 111,
) -> dict[str, float | int]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_scores_array = np.asarray(y_scores, dtype=float)

    if y_true_array.size == 0:
        raise ValueError("Bootstrap requires a non-empty test set.")

    rng = np.random.default_rng(random_state)
    auc_scores: list[float] = []
    f1_scores: list[float] = []
    attempts = 0
    max_attempts = n_bootstraps * 25

    while len(auc_scores) < n_bootstraps and attempts < max_attempts:
        sample_indices = rng.integers(0, y_true_array.size, size=y_true_array.size)
        sampled_y = y_true_array[sample_indices]
        sampled_scores = y_scores_array[sample_indices]

        attempts += 1
        if np.unique(sampled_y).size < 2:
            continue

        sampled_predictions = (sampled_scores >= threshold).astype(int)
        auc_scores.append(float(roc_auc_score(sampled_y, sampled_scores)))
        f1_scores.append(
            float(
                f1_score(
                    sampled_y,
                    sampled_predictions,
                    pos_label=1,
                    zero_division=0,
                )
            )
        )

    if not auc_scores:
        raise ValueError("Unable to compute bootstrap intervals with the available test labels.")

    auc_array = np.asarray(auc_scores, dtype=float)
    f1_array = np.asarray(f1_scores, dtype=float)

    return {
        "bootstrap_samples": int(auc_array.size),
        "auc_roc_ci_lower": float(np.percentile(auc_array, 2.5)),
        "auc_roc_ci_upper": float(np.percentile(auc_array, 97.5)),
        "f1_ci_lower": float(np.percentile(f1_array, 2.5)),
        "f1_ci_upper": float(np.percentile(f1_array, 97.5)),
    }
