from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cold_start_replenishment.paths import resolve_repo_path


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: Path
    filenames: dict[str, str] = Field(default_factory=dict)
    target_date_columns: list[str] = Field(default_factory=list)
    hierarchy_fields: list[str] = Field(default_factory=list)


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    random_seed: int = 20260805
    audit_output_path: Path
    minimum_active_history: int
    minimum_evaluation_horizon: int
    minimum_nonzero_observations: int
    maximum_sampled_series: int
    onset_activity_window: int = 28
    minimum_donor_candidates: int = 20
    m5: DatasetConfig
    car_parts: DatasetConfig
    spdf: DatasetConfig
    aviation_synthetic: DatasetConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in configuration: {resolved}")
    return value


def load_audit_config(path: str | Path = "configs/dataset_audit.yaml") -> AuditConfig:
    return AuditConfig.model_validate(load_yaml(path))
