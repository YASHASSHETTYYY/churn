from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(config_path: str | Path = "params.yaml") -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    resolved_path = resolve_config_path(config_path)
    with resolved_path.open(encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


def resolve_path(
    path_value: str | Path,
    config_path: str | Path = "params.yaml",
) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    config_dir = resolve_config_path(config_path).parent
    return (config_dir / path).resolve()
