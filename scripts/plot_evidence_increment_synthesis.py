#!/usr/bin/env python3
"""Generate the cross-dataset evidence-increment forest plot from frozen rows."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "scripts")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from audit_panel_alignment import require_matplotlib_panel_alignment

DEFAULT_CONFIG = ROOT / "configs/evidence_increment_synthesis.json"
DEFAULT_OUTPUT = ROOT / "outputs/ai_darld_v3/evidence_increment_synthesis"
DEFAULT_FIGURE = DEFAULT_OUTPUT / "evidence_increment_forest.pdf"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_difference(first: np.ndarray, comparator: np.ndarray) -> float:
    denominator = float(np.mean(comparator))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Comparator mean cost must be positive and finite")
    return 100.0 * (float(np.mean(first)) - denominator) / denominator


def validate_and_pair(
    rows: pd.DataFrame,
    *,
    key_columns: list[str],
    method_column: str,
    cost_column: str,
    actual_column: str | None,
    first_method: str,
    comparator_method: str,
) -> pd.DataFrame:
    required = set(key_columns + [method_column, cost_column])
    if actual_column is not None:
        required.add(actual_column)
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    selected = rows[rows[method_column].isin([first_method, comparator_method])].copy()
    duplicates = selected.duplicated(key_columns + [method_column], keep=False)
    if duplicates.any():
        sample = selected.loc[duplicates, key_columns + [method_column]].head()
        raise ValueError(f"Duplicate method rows detected:\n{sample}")
    counts = selected.groupby(method_column, observed=True).size().to_dict()
    if counts.get(first_method, 0) == 0 or counts.get(comparator_method, 0) == 0:
        raise ValueError("One or both requested policies are absent")
    first = selected[selected[method_column].eq(first_method)].set_index(key_columns)
    comparator = selected[selected[method_column].eq(comparator_method)].set_index(key_columns)
    if not first.index.is_unique or not comparator.index.is_unique:
        raise ValueError("Comparison keys are not unique")
    missing_first = comparator.index.difference(first.index)
    missing_comparator = first.index.difference(comparator.index)
    if len(missing_first) or len(missing_comparator):
        raise ValueError(
            "Unmatched evaluation keys: "
            f"missing_first={len(missing_first)}, "
            f"missing_comparator={len(missing_comparator)}"
        )
    if len(first) != len(comparator):
        raise ValueError("Matched policies have different row counts")
    if actual_column is not None and not np.array_equal(
        first.loc[comparator.index, actual_column].to_numpy(),
        comparator[actual_column].to_numpy(),
        equal_nan=True,
    ):
        raise ValueError("Matched policies do not share identical realized demand")
    paired = pd.DataFrame(
        {
            "first_cost": first.loc[comparator.index, cost_column].astype(float),
            "comparator_cost": comparator[cost_column].astype(float),
        },
        index=comparator.index,
    ).reset_index()
    if paired[["first_cost", "comparator_cost"]].isna().any().any():
        raise ValueError("Matched costs contain missing values")
    return paired


def clustered_bootstrap(
    paired: pd.DataFrame,
    *,
    product_column: str,
    draws: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    grouped = paired.groupby(product_column, sort=True, observed=True).agg(
        first_sum=("first_cost", "sum"),
        comparator_sum=("comparator_cost", "sum"),
        row_count=("first_cost", "size"),
    )
    if grouped.empty:
        raise ValueError("No products available for bootstrap")
    point = relative_difference(
        paired["first_cost"].to_numpy(), paired["comparator_cost"].to_numpy()
    )
    rng = np.random.default_rng(seed)
    n_products = len(grouped)
    sampled = rng.integers(0, n_products, size=(draws, n_products))
    first_sum = grouped["first_sum"].to_numpy()[sampled].sum(axis=1)
    comparator_sum = grouped["comparator_sum"].to_numpy()[sampled].sum(axis=1)
    row_count = grouped["row_count"].to_numpy()[sampled].sum(axis=1)
    first_mean = first_sum / row_count
    comparator_mean = comparator_sum / row_count
    if np.any(comparator_mean <= 0):
        raise ValueError("A bootstrap comparator mean is nonpositive")
    estimates = 100.0 * (first_mean - comparator_mean) / comparator_mean
    return point, estimates


def clustered_bootstrap_product_sums(
    paired: pd.DataFrame, *, draws: int, seed: int
) -> tuple[float, np.ndarray]:
    """Bootstrap de-identified product cost sums while retaining row weights."""
    if paired.empty:
        raise ValueError("No products available for bootstrap")
    first_total = float(paired["first_sum"].sum())
    comparator_total = float(paired["comparator_sum"].sum())
    row_total = float(paired["row_count"].sum())
    point = 100.0 * (first_total / row_total - comparator_total / row_total) / (
        comparator_total / row_total
    )
    rng = np.random.default_rng(seed)
    n_products = len(paired)
    sampled = rng.integers(0, n_products, size=(draws, n_products))
    first_sum = paired["first_sum"].to_numpy()[sampled].sum(axis=1)
    comparator_sum = paired["comparator_sum"].to_numpy()[sampled].sum(axis=1)
    row_count = paired["row_count"].to_numpy()[sampled].sum(axis=1)
    first_mean = first_sum / row_count
    comparator_mean = comparator_sum / row_count
    if np.any(comparator_mean <= 0):
        raise ValueError("A bootstrap comparator mean is nonpositive")
    return point, 100.0 * (first_mean - comparator_mean) / comparator_mean


def load_compact_pairs(
    compact: pd.DataFrame, *, dataset: str, first_policy: str, comparator: str
) -> pd.DataFrame:
    rows = compact[
        compact["dataset"].eq(dataset)
        & compact["policy"].isin([first_policy, comparator])
    ].copy()
    if rows.duplicated(["redacted_product_id", "policy"]).any():
        raise ValueError(f"{dataset}: duplicate compact product-policy rows")
    order_counts = rows.groupby("redacted_product_id", observed=True)["bootstrap_order"].nunique()
    if not order_counts.eq(1).all():
        raise ValueError(f"{dataset}: inconsistent bootstrap order")
    order = rows.drop_duplicates("redacted_product_id").set_index("redacted_product_id")[
        "bootstrap_order"
    ]
    pivot_sum = rows.pivot(
        index="redacted_product_id", columns="policy", values="cost_sum"
    )
    pivot_count = rows.pivot(
        index="redacted_product_id", columns="policy", values="context_row_count"
    )
    if set(pivot_sum.columns) != {first_policy, comparator} or pivot_sum.isna().any().any():
        raise ValueError(f"{dataset}: incomplete compact policy pairing")
    if not pivot_count[first_policy].eq(pivot_count[comparator]).all():
        raise ValueError(f"{dataset}: repeated contexts differ between policies")
    result = pd.DataFrame(
        {
            "redacted_product_id": pivot_sum.index,
            "first_sum": pivot_sum[first_policy].to_numpy(float),
            "comparator_sum": pivot_sum[comparator].to_numpy(float),
            "row_count": pivot_count[first_policy].to_numpy(int),
        }
    )
    result["bootstrap_order"] = result["redacted_product_id"].map(order)
    return result.sort_values("bootstrap_order").reset_index(drop=True)


def compute_contrasts(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    bootstrap_records: list[dict[str, Any]] = []
    draws = int(config["bootstrap_draws"])
    seed = int(config["bootstrap_seed"])
    compact_source = ROOT / config["compact_source"]
    compact = pd.read_csv(compact_source)
    required = {
        "dataset", "redacted_product_id", "bootstrap_order", "policy", "cost_sum",
        "context_row_count",
    }
    if not required.issubset(compact.columns):
        raise ValueError(f"Compact source lacks columns: {sorted(required - set(compact.columns))}")
    for panel in config["panel_order"]:
        panel_spec = config["panels"][panel]
        for dataset in config["dataset_order"]:
            spec = config["datasets"][dataset]
            first_label = panel_spec["first_policy"]
            comparator_label = panel_spec["comparator"]
            paired = load_compact_pairs(
                compact, dataset=dataset, first_policy=first_label, comparator=comparator_label
            )
            point, estimates = clustered_bootstrap_product_sums(paired, draws=draws, seed=seed)
            low, high = np.quantile(estimates, [0.025, 0.975])
            source = compact_source
            records.append(
                {
                    "panel": panel,
                    "increment": panel_spec["increment"],
                    "dataset": dataset,
                    "first_policy": first_label,
                    "comparator": comparator_label,
                    "point_estimate_percent": point,
                    "ci_low_percent": low,
                    "ci_high_percent": high,
                    "products": len(paired),
                    "matched_rows": int(paired["row_count"].sum()),
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                    "source_file": source.relative_to(ROOT).as_posix(),
                    "source_protocol": spec["source_protocol"],
                }
            )
            bootstrap_records.append(
                {
                    "panel": panel,
                    "dataset": dataset,
                    "draw_mean_percent": float(np.mean(estimates)),
                    "draw_sd_percent": float(np.std(estimates, ddof=1)),
                    "draw_min_percent": float(np.min(estimates)),
                    "draw_max_percent": float(np.max(estimates)),
                    "draw_sha256": hashlib.sha256(
                        estimates.astype("<f8", copy=False).tobytes()
                    ).hexdigest(),
                }
            )
    contrasts = pd.DataFrame.from_records(records)
    expected = len(config["panel_order"]) * len(config["dataset_order"])
    if len(contrasts) != expected:
        raise ValueError(f"Expected {expected} contrasts, found {len(contrasts)}")
    coverage = contrasts.groupby("panel", observed=True)["dataset"].nunique()
    if not (coverage == len(config["dataset_order"])).all():
        raise ValueError("Every panel must contain all configured datasets")
    return contrasts, pd.DataFrame.from_records(bootstrap_records)


def plot_forest(contrasts: pd.DataFrame, config: dict[str, Any], path: Path) -> None:
    panel_titles = {
        "A": ("Multi-donor aggregation", "Complete similarity − single donor"),
        "B": ("Learned adaptation", "Component-specific − complete similarity"),
        "C": ("Component configuration", "Component-specific − matched shared"),
    }
    datasets = config["dataset_order"]
    low = float(contrasts["ci_low_percent"].min())
    high = float(contrasts["ci_high_percent"].max())
    bound = max(abs(low), abs(high)) + 2.0
    xlim = (-bound, bound)
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.05), sharex=True, sharey=True)
    color = "#315f7d"
    for axis, panel in zip(axes, config["panel_order"], strict=True):
        data = contrasts[contrasts["panel"].eq(panel)].set_index("dataset").loc[datasets]
        y = np.arange(len(datasets))[::-1]
        point = data["point_estimate_percent"].to_numpy()
        xerr = np.vstack(
            [
                point - data["ci_low_percent"].to_numpy(),
                data["ci_high_percent"].to_numpy() - point,
            ]
        )
        for row in y:
            axis.hlines(row, *xlim, color="#e5e9ed", linewidth=0.45, zorder=0)
        axis.axvline(0, color="#78828c", linewidth=0.75, zorder=1)
        axis.errorbar(
            point,
            y,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor="#3f4850",
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.75,
            markersize=3.8,
            markeredgewidth=0.4,
            zorder=3,
        )
        title, subtitle = panel_titles[panel]
        axis.text(-0.16, 1.18, panel.lower(), transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=7.2, fontweight="bold", color="#25364a")
        axis.text(0.0, 1.18, title, transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=7.0, fontweight="bold", color="#25364a")
        axis.text(0.0, 1.03, subtitle, transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=5.7, color="#596777")
        axis.set_xlim(*xlim)
        axis.set_yticks(y, datasets)
        axis.set_xticks([-30, -15, 0, 15, 30])
        axis.tick_params(axis="x", labelsize=6.0, length=2.2, width=0.55,
                         color="#78828c", pad=1.5)
        axis.tick_params(axis="y", labelsize=6.3, length=0, pad=3)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#9aa3ab")
        axis.spines["bottom"].set_linewidth(0.55)
    fig.supxlabel("Relative mean-cost difference (%)", fontsize=6.5, y=0.012,
                  color="#334155")
    fig.subplots_adjust(left=0.19, right=0.99, top=0.94, bottom=0.10, hspace=0.58)
    require_matplotlib_panel_alignment(
        fig,
        json_out=path.parent / "evidence_increment_forest.alignment.json",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Creator": "cold-start-replenishment evidence synthesis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    svg_path = path.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    figure = args.figure.resolve()
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    contrasts, bootstrap_summary = compute_contrasts(config)
    contrast_path = output / "relative_cost_contrasts.csv"
    bootstrap_path = output / "bootstrap_draw_summary.csv"
    contrasts.to_csv(contrast_path, index=False, float_format="%.10f")
    bootstrap_summary.to_csv(bootstrap_path, index=False, float_format="%.10f")
    plot_forest(contrasts, config, figure)
    manifest = {
        "schema": config["schema"],
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "generator": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "bootstrap_draws": config["bootstrap_draws"],
        "bootstrap_seed": config["bootstrap_seed"],
        "estimand": config["estimand"],
        "interval": config["interval"],
        "sign_convention": config["sign_convention"],
        "compact_source": config["compact_source"],
        "source_sha256": {config["compact_source"]: sha256(ROOT / config["compact_source"])},
        "outputs": {
            contrast_path.name: sha256(contrast_path),
            bootstrap_path.name: sha256(bootstrap_path),
            figure.relative_to(ROOT).as_posix(): sha256(figure),
            figure.with_suffix(".png").relative_to(ROOT).as_posix(): sha256(
                figure.with_suffix(".png")
            ),
            figure.with_suffix(".svg").relative_to(ROOT).as_posix(): sha256(
                figure.with_suffix(".svg")
            ),
            "evidence_increment_forest.alignment.json": sha256(
                figure.parent / "evidence_increment_forest.alignment.json"
            ),
        },
        "coverage": {
            "panels": config["panel_order"],
            "datasets": config["dataset_order"],
            "contrasts": len(contrasts),
        },
        "figure_contract": {
            "core_conclusion": "The evaluations separate aggregation, adaptation, and component-configuration increments without implying a universal ranking.",
            "results_question": "Which controlled increment is supported in each evaluation population?",
            "archetype": "compact quantitative grid",
            "backend": "Python/Matplotlib",
            "final_size_inches": [3.35, 4.05],
            "panel_map": {
                "A": "multi-donor aggregation",
                "B": "learned adaptation",
                "C": "component-specific configuration",
            },
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
