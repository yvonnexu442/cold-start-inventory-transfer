#!/usr/bin/env python3
"""Generate the frozen MAN/BRAF checkpoint inputs from the source workbooks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cold_start_replenishment.data.spdf_pilot import parse_braf, parse_man  # noqa: E402
from cold_start_replenishment.evaluation.sprint2 import _run_dataset  # noqa: E402


def _load_config() -> dict[str, Any]:
    path = ROOT / "configs/full_scale.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN BEFORE FULL-SCALE EXECUTION":
        raise ValueError(f"Unexpected full-scale configuration status in {path}")
    return config


def generate(dataset_name: str) -> dict[str, Any]:
    """Run the frozen source-data preparation and write all checkpoint frames."""
    config = _load_config()
    dataset = parse_man() if dataset_name == "MAN" else parse_braf()
    started = time.perf_counter()
    results, reliability, calibration, cutoffs, scenarios = _run_dataset(dataset, config)
    destination = ROOT / "outputs/full_scale/checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {
        "results": results,
        "reliability": reliability,
        "calibration": calibration,
        "cutoffs": cutoffs,
        "scenarios": scenarios,
    }
    for label, frame in frames.items():
        frame.to_parquet(destination / f"{dataset_name.lower()}_{label}.parquet", index=False)
    status = {
        "dataset": dataset_name,
        "configuration": "configs/full_scale.yaml",
        "targets": int(results["target_id"].nunique()),
        "cutoffs": int(results["cutoff"].nunique()),
        "target_cutoff_rows": int(
            results[["target_id", "cutoff"]].drop_duplicates().shape[0]
        ),
        "operational_rows": int(len(results)),
        "runtime_seconds": time.perf_counter() - started,
        "completed": True,
    }
    (destination / f"{dataset_name.lower()}_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate frozen service-parts checkpoints from raw MAN/BRAF data."
    )
    parser.add_argument("--dataset", choices=("MAN", "BRAF", "all"), default="all")
    args = parser.parse_args()
    names = ("MAN", "BRAF") if args.dataset == "all" else (args.dataset,)
    for name in names:
        print(json.dumps(generate(name), indent=2))


if __name__ == "__main__":
    main()
