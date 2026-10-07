#!/usr/bin/env python3
"""Create prespecified aggregate evidence from the frozen Favorita run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _cluster_interval(values: pd.Series, draws: int, seed: int) -> tuple[float, float]:
    array = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = np.asarray(
        [rng.choice(array, len(array), replace=True).mean() for _ in range(draws)]
    )
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/ai_darld_v3/favorita_confirmation_v1/row_level_results.parquet",
    )
    parser.add_argument(
        "--seed-input",
        default="outputs/ai_darld_v3/favorita_confirmation_v1/policy_seed_rows.parquet",
    )
    parser.add_argument(
        "--output-dir", default="outputs/ai_darld_v3/favorita_confirmation_v1"
    )
    args = parser.parse_args()
    rows = pd.read_parquet(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    context = (
        rows.groupby(["method", "cutoff", "horizon", "cost_ratio"], as_index=False)
        .agg(
            products=("target_id", "nunique"),
            cost=("cost", "mean"),
            service=("service", "mean"),
            pinball=("pinball", "mean"),
            brier=("brier", "mean"),
            mean_action=("action", "mean"),
        )
    )
    context.to_csv(output / "context_summary.csv", index=False)

    keys = ["target_id", "cutoff", "horizon", "cost_ratio"]
    cost = rows.pivot(index=keys, columns="method", values="cost")
    service = rows.pivot(index=keys, columns="method", values="service")
    base = "component_specific_transfer"
    contrasts = []
    for method in sorted(rows.method.unique()):
        product_cost = (cost[base] - cost[method]).groupby(level=0).mean()
        product_service = (service[base] - service[method]).groupby(level=0).mean()
        cost_low, cost_high = _cluster_interval(product_cost, 5000, 20261004)
        service_low, service_high = _cluster_interval(product_service, 5000, 20261005)
        comparator_mean = float(rows.loc[rows.method.eq(method), "cost"].mean())
        contrasts.append(
            {
                "comparator": method,
                "component_specific_minus_comparator_cost": float(product_cost.mean()),
                "relative_cost_difference": float(product_cost.mean() / comparator_mean),
                "cost_ci_low": cost_low,
                "cost_ci_high": cost_high,
                "component_specific_minus_comparator_service": float(product_service.mean()),
                "service_ci_low": service_low,
                "service_ci_high": service_high,
                "products": int(len(product_cost)),
            }
        )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_path = output / "paired_contrasts.csv"
    contrast_frame.to_csv(contrast_path, index=False)

    # paired_contrasts.csv is the paper's paired-CI authority.  Synchronize the
    # duplicate columns in summary.csv rather than drawing from a second RNG stream.
    summary_path = output / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path).drop(
            columns=["component_specific_minus_method", "ci_low", "ci_high", "products"],
            errors="ignore",
        )
        authoritative = contrast_frame.rename(
            columns={
                "comparator": "method",
                "component_specific_minus_comparator_cost": "component_specific_minus_method",
                "cost_ci_low": "ci_low",
                "cost_ci_high": "ci_high",
            }
        )[["method", "component_specific_minus_method", "ci_low", "ci_high", "products"]]
        summary.merge(authoritative, on="method", how="left", validate="one_to_one").to_csv(
            summary_path, index=False
        )

    seed_rows = pd.read_parquet(args.seed_input)
    seed_summary = (
        seed_rows.groupby(["method", "policy_seed"], as_index=False)
        .agg(cost=("cost", "mean"), service=("service", "mean"), brier=("event_probability", lambda p: float(np.mean((p - (seed_rows.loc[p.index, "actual"] > 0).astype(float)) ** 2))))
    )
    seed_summary.to_csv(output / "policy_seed_summary.csv", index=False)

    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["paired_interval_authority"] = {
            "file": contrast_path.as_posix(),
            "bootstrap_draws": 5000,
            "cost_seed": 20261004,
            "service_seed": 20261005,
            "seed_scope": "reinitialized independently for each comparator",
            "sha256": _sha256(contrast_path),
        }
        manifest["summary_sha256"] = _sha256(summary_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
