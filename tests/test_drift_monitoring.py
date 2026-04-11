from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.monitoring import drift_report
from src.monitoring.drift_injector import (
    create_drift_scenario,
    shift_categorical_distribution,
    shift_numeric_feature,
)
from src.monitoring.drift_report import EvidentlyDriftError, generate_drift_report
from src.monitoring.psi_detector import compute_psi, detect_psi_drift


def sample_drift_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_day_minutes": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "international_plan": ["no", "no", "yes", "no", "no", "yes"],
            "number_customer_service_calls": [1, 1, 2, 2, 3, 1],
        }
    )


def test_shift_numeric_feature_moves_mean():
    frame = sample_drift_frame()
    shifted = shift_numeric_feature(
        frame,
        feature="total_day_minutes",
        shift_amount=25.0,
        noise_std=0.0,
        random_state=7,
    )

    assert shifted["total_day_minutes"].mean() == pytest.approx(
        frame["total_day_minutes"].mean() + 25.0
    )


def test_shift_categorical_distribution_reaches_target_proportion():
    frame = sample_drift_frame()
    shifted = shift_categorical_distribution(
        frame,
        feature="international_plan",
        target_category="yes",
        new_proportion=0.5,
        random_state=7,
    )

    assert (shifted["international_plan"] == "yes").mean() == pytest.approx(0.5, abs=0.17)


def test_create_drift_scenario_returns_expected_metadata():
    frame = sample_drift_frame()
    drifted, metadata = create_drift_scenario(frame, scenario="seasonal", magnitude=0.2, random_state=11)

    assert metadata.scenario == "seasonal"
    assert metadata.feature_drifted == "number_customer_service_calls"
    assert not drifted.equals(frame)


def test_compute_psi_flags_large_distribution_shift():
    expected = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 10)
    shifted = expected + 20

    stable_score = compute_psi(expected, expected)
    shifted_score = compute_psi(expected, shifted)
    detection = detect_psi_drift(expected, shifted, threshold=0.2)

    assert stable_score < 0.05
    assert shifted_score > 0.2
    assert detection["detected"] is True


def test_generate_drift_report_raises_when_evidently_fails(monkeypatch):
    runtime_root = Path.cwd() / "test_artifacts_drift_monitoring"
    runtime_root.mkdir(parents=True, exist_ok=True)
    config = {
        "drift_monitoring": {
            "report_html": str(runtime_root / "reports" / "drift_report.html"),
            "summary_json": str(runtime_root / "reports" / "drift_report.json"),
        }
    }
    config_path = runtime_root / "params.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(
        drift_report,
        "_build_monitoring_frames",
        lambda config_path="params.yaml": (
            pd.DataFrame({"feature": [1.0, 2.0]}),
            pd.DataFrame({"feature": [1.5, 2.5]}),
            None,
            ["feature"],
        ),
    )
    monkeypatch.setattr(
        drift_report,
        "_load_evidently_api",
        lambda: (_ for _ in ()).throw(EvidentlyDriftError("mocked Evidently failure")),
    )

    with pytest.raises(EvidentlyDriftError, match="mocked Evidently failure"):
        generate_drift_report(config_path=config_path)

    assert not Path(config["drift_monitoring"]["summary_json"]).exists()
