#!/usr/bin/env python3
"""Run the legacy-named, pre-specified Favorita external evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cold_start_replenishment.analogs.direct_mixture_learning import (  # noqa: E402
    MixtureDecisionTask,
)
from cold_start_replenishment.analogs.factorized_transfer import (  # noqa: E402
    FactorizedTransferPolicy,
    complete_mixture_distribution,
    factorized_distribution,
)
from cold_start_replenishment.data.favorita import (  # noqa: E402
    FavoritaPanel,
    frozen_item_roles,
    load_favorita_panel,
    stable_item_hash,
)
from cold_start_replenishment.inventory.newsvendor import (  # noqa: E402
    optimal_feasible_level,
)


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_panel(panel: FavoritaPanel, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "item_ids.npy", panel.item_ids)
    np.save(cache / "dates.npy", panel.dates.to_numpy(dtype="datetime64[D]"))
    np.save(cache / "quantities.npy", panel.quantities)
    panel.metadata.to_parquet(cache / "metadata.parquet", index=False)
    panel.store_diagnostic.to_csv(cache / "store_aggregation_diagnostic.csv", index=False)
    (cache / "negative_sales_diagnostic.json").write_text(
        json.dumps(panel.negative_diagnostic, indent=2) + "\n"
    )


def _load_panel(cache: Path) -> FavoritaPanel:
    quantities = np.load(cache / "quantities.npy", mmap_mode="r")
    return FavoritaPanel(
        item_ids=np.load(cache / "item_ids.npy"),
        dates=pd.DatetimeIndex(np.load(cache / "dates.npy")),
        quantities=quantities,
        metadata=pd.read_parquet(cache / "metadata.parquet"),
        store_diagnostic=pd.read_csv(cache / "store_aggregation_diagnostic.csv"),
        negative_diagnostic=json.loads((cache / "negative_sales_diagnostic.json").read_text()),
    )


def _prepare_panel(cfg: dict, *, force: bool = False) -> FavoritaPanel:
    cache = ROOT / "data/processed/favorita_confirmation_v1"
    if not force and (cache / "quantities.npy").exists():
        return _load_panel(cache)
    raw = ROOT / "data/raw/favorita/extracted_v1/csv"
    panel = load_favorita_panel(
        raw,
        calendar_start=cfg["calendar_and_demand"]["observation_calendar_start"],
        calendar_end=cfg["calendar_and_demand"]["observation_calendar_end"],
    )
    _save_panel(panel, cache)
    return panel


def _cutoff_index(dates: pd.DatetimeIndex, cutoff: str) -> int:
    matches = np.flatnonzero(dates == pd.Timestamp(cutoff))
    if len(matches) != 1:
        raise ValueError(f"cutoff {cutoff} is not uniquely present")
    return int(matches[0] + 1)


def _history(series: np.ndarray, cutoff: int, days: int) -> np.ndarray:
    if cutoff < days:
        raise ValueError("cutoff does not have the frozen history length")
    return np.asarray(series[cutoff - days : cutoff], float)


def _horizon_atoms(history: np.ndarray, horizon: int, count: int) -> np.ndarray:
    if len(history) < horizon:
        raise ValueError("history is shorter than the demand horizon")
    windows = np.convolve(history, np.ones(horizon), mode="valid")
    positions = (np.arange(count) + 0.5) / count
    return np.quantile(windows, positions, method="nearest").astype(float)


def _features(similarity: np.ndarray, scenarios: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
    event = np.mean(scenarios > 0, axis=1)
    positive_support = np.asarray([np.count_nonzero(row > 0) for row in histories], float)
    simulation_positive = np.sum(scenarios > 0, axis=1)
    positive_mean = np.divide(
        scenarios.sum(axis=1),
        simulation_positive,
        out=np.zeros(len(scenarios)),
        where=simulation_positive > 0,
    )
    return np.column_stack([similarity, event, positive_support, positive_mean])


def _support(histories: list[np.ndarray], horizon: int, atoms: int) -> np.ndarray:
    return np.asarray(
        [
            [len(row), np.count_nonzero(row > 0), len(row) - horizon + 1, atoms]
            for row in histories
        ],
        float,
    )


def _fit_metadata(panel: FavoritaPanel, development: np.ndarray):
    columns = ["family", "class", "perishable"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    encoder.fit(panel.metadata.iloc[development][columns])
    encoded = normalize(encoder.transform(panel.metadata[columns]), norm="l2")
    dense = encoded.toarray()
    return encoded, dense, encoder


def _eligible_donors(
    quantities: np.ndarray,
    development: np.ndarray,
    cutoff: int,
    history_days: int,
    minimum_positive: int,
) -> np.ndarray:
    visible = np.asarray(quantities[development, cutoff - history_days : cutoff])
    keep = np.count_nonzero(visible > 0, axis=1) >= minimum_positive
    return development[keep]


def _nearest(encoded, target: int, donors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(encoded[target].dot(encoded[donors].T).todense()).ravel()
    order = np.lexsort((donors, -scores))[:k]
    return donors[order], np.clip(scores[order], 0.0, None)


def _make_task(
    target: int,
    donors: np.ndarray,
    similarities: np.ndarray,
    quantities: np.ndarray,
    cutoff: int,
    horizon: int,
    ratio: float,
    history_days: int,
    atoms: int,
) -> MixtureDecisionTask:
    histories = [_history(quantities[index], cutoff, history_days) for index in donors]
    scenarios = np.vstack([_horizon_atoms(row, horizon, atoms) for row in histories])
    actual = float(np.asarray(quantities[target, cutoff : cutoff + horizon], float).sum())
    return MixtureDecisionTask(
        str(target),
        _features(similarities, scenarios, histories),
        scenarios,
        actual,
        ratio,
        horizon,
        None,
        None,
        1.0,
        0.0,
        _support(histories, horizon, atoms),
    )


def _pseudo_targets(
    indices: np.ndarray,
    item_ids: np.ndarray,
    salt: str,
    count: int,
) -> np.ndarray:
    ordered = sorted(
        map(int, indices),
        key=lambda index: (
            stable_item_hash(int(item_ids[index]), salt),
            int(item_ids[index]),
        ),
    )
    return np.asarray(ordered[:count], int)


def _build_policy_tasks(
    indices: np.ndarray,
    development: np.ndarray,
    encoded,
    quantities: np.ndarray,
    cutoff_indices: list[int],
    cfg: dict,
) -> list[MixtureDecisionTask]:
    tasks: list[MixtureDecisionTask] = []
    task_cfg = cfg["tasks"]
    for cutoff in cutoff_indices:
        eligible = _eligible_donors(
            quantities,
            development,
            cutoff,
            int(task_cfg["required_history_days"]),
            int(cfg["eligibility"]["donor_minimum_positive_history_days"]),
        )
        for target in map(int, indices):
            pool = eligible[eligible != target]
            donors, similarity = _nearest(
                encoded, target, pool, int(task_cfg["donors_per_target"])
            )
            for horizon in task_cfg["demand_horizons_days"]:
                for ratio in task_cfg["shortage_holding_ratios"]:
                    tasks.append(
                        _make_task(
                            target,
                            donors,
                            similarity,
                            quantities,
                            cutoff,
                            int(horizon),
                            float(ratio),
                            int(task_cfg["required_history_days"]),
                            int(task_cfg["policy_training_atoms"]),
                        )
                    )
    return tasks


def _ensemble_distribution(task: MixtureDecisionTask, policies: list[FactorizedTransferPolicy]):
    values, masses, events = [], [], []
    for policy in policies:
        value, mass, diagnostic = factorized_distribution(
            task,
            policy.parameters_,
            policy.occurrence_logit_shift_,
            shared=policy.shared,
            contraction=policy.contraction,
        )
        values.extend(value.tolist())
        masses.extend((mass / len(policies)).tolist())
        events.append(diagnostic["event_probability"])
    return np.asarray(values), np.asarray(masses), float(np.mean(events))


def _rolling_rows(
    dense: np.ndarray,
    quantities: np.ndarray,
    development: np.ndarray,
    cutoff: int,
    horizon: int,
    origins: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    for step in range(1, origins + 1):
        start = cutoff - step * horizon
        end = start + horizon
        if start < 0 or end > cutoff:
            raise ValueError("rolling origin violates the evaluation cutoff")
        x_rows.append(dense[development])
        y_rows.append(np.asarray(quantities[development, start:end], float).sum(axis=1))
    return np.vstack(x_rows), np.concatenate(y_rows)


@dataclass
class _ConstantClassifier:
    probability: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        p = np.full(len(values), self.probability)
        return np.column_stack([1.0 - p, p])


def _fit_global_models(dense, quantities, development, cutoff_indices, cfg):
    models = {}
    quantiles = cfg["global_quantile"]["quantiles"]
    origins = int(cfg["global_quantile"]["rolling_origins_per_development_product"])
    for cutoff_label, cutoff in zip(cfg["tasks"]["cutoffs"], cutoff_indices, strict=True):
        for horizon in cfg["tasks"]["demand_horizons_days"]:
            xx, yy = _rolling_rows(dense, quantities, development, cutoff, int(horizon), origins)
            for tau in quantiles:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(tau),
                    max_iter=100,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    learning_rate=0.05,
                    random_state=20261004,
                ).fit(xx, yy)
                models[(str(cutoff_label), int(horizon), round(float(tau), 7))] = model
    return models


def _fit_zig_models(dense, quantities, development, cutoff_indices, cfg):
    models = {}
    origins = int(cfg["global_quantile"]["rolling_origins_per_development_product"])
    for cutoff_label, cutoff in zip(cfg["tasks"]["cutoffs"], cutoff_indices, strict=True):
        for horizon in cfg["tasks"]["demand_horizons_days"]:
            xx, yy = _rolling_rows(dense, quantities, development, cutoff, int(horizon), origins)
            fitted = []
            event = (yy > 0).astype(int)
            for seed in cfg["methods"]["seeds"]:
                common = dict(
                    iterations=100,
                    depth=5,
                    learning_rate=0.05,
                    random_seed=int(seed),
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                )
                classifier = (
                    _ConstantClassifier(float(event[0]))
                    if np.unique(event).size == 1
                    else CatBoostClassifier(loss_function="Logloss", **common).fit(xx, event)
                )
                positive = yy > 0
                if not np.any(positive):
                    raise ValueError("adapted ZIG-MC has no positive development labels")
                regressor = CatBoostRegressor(loss_function="RMSE", **common).fit(
                    xx[positive], np.log1p(yy[positive])
                )
                fitted.append((classifier, regressor))
            models[(str(cutoff_label), int(horizon))] = fitted
    return models


def _zig_distribution(models, xrow: np.ndarray, seed: int):
    probabilities, means = [], []
    for classifier, regressor in models:
        probabilities.append(float(classifier.predict_proba(xrow[None])[0, 1]))
        means.append(max(float(np.expm1(regressor.predict(xrow[None])[0])), 0.0))
    probability = float(np.mean(probabilities))
    rng = np.random.default_rng(seed)
    values, masses = [0.0], [1.0 - probability]
    for mean in means:
        samples = rng.gamma(2.0, max(mean, 1e-8) / 2.0, 80)
        values.extend(samples.tolist())
        masses.extend(np.full(80, probability / (80 * len(means))).tolist())
    return np.asarray(values), np.asarray(masses), probability


def _score(action: float, actual: float, ratio: float) -> tuple[float, float, float]:
    cost = max(action - actual, 0.0) + ratio * max(actual - action, 0.0)
    service = 1.0 if actual <= 0 else min(action, actual) / actual
    tau = ratio / (1.0 + ratio)
    pinball = (tau - float(actual < action)) * (actual - action)
    return float(cost), float(service), float(pinball)


def _bootstrap(rows: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    pivot = rows.pivot_table(
        index=["target_id", "cutoff", "horizon", "cost_ratio"],
        columns="method",
        values="cost",
    )
    if pivot.isna().any().any():
        raise ValueError("missing method rows on matched evaluation keys")
    by_target = pivot.groupby(level=0).mean()
    base = "component_specific_transfer"
    output = []
    for method in sorted(rows.method.unique()):
        delta = (by_target[base] - by_target[method]).to_numpy()
        # A fresh generator makes each comparator's interval independent of
        # method iteration order and identical to the paired-CI authority.
        rng = np.random.default_rng(seed)
        samples = np.asarray(
            [rng.choice(delta, len(delta), replace=True).mean() for _ in range(draws)]
        )
        output.append(
            {
                "method": method,
                "component_specific_minus_method": float(delta.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "products": int(len(delta)),
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/favorita_untouched_confirmation_v1.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    started = time.time()
    panel = _prepare_panel(cfg, force=args.force_prepare)
    output = ROOT / "outputs/ai_darld_v3/favorita_confirmation_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "negative_sales_diagnostic.json").write_text(
        json.dumps(panel.negative_diagnostic, indent=2) + "\n"
    )
    panel.store_diagnostic.to_csv(output / "store_aggregation_diagnostic.csv", index=False)
    if args.prepare_only:
        print(json.dumps(panel.negative_diagnostic, indent=2))
        return

    item_to_index = {int(item): index for index, item in enumerate(panel.item_ids)}
    role_items = frozen_item_roles(
        panel.item_ids,
        salt=cfg["roles"]["salt"],
        confirmation_count=int(cfg["roles"]["confirmation_target_count"]),
    )
    roles = {name: np.asarray([item_to_index[int(item)] for item in values], int) for name, values in role_items.items()}
    if any(set(role_items[a]) & set(role_items[b]) for a, b in (("development", "validation"), ("development", "confirmation"), ("validation", "confirmation"))):
        raise ValueError("role overlap")
    role_manifest = pd.concat(
        [pd.DataFrame({"item_nbr": role_items[name], "role": name}) for name in roles],
        ignore_index=True,
    )
    role_manifest.to_csv(output / "split_manifest.csv", index=False)

    cutoff_indices = [_cutoff_index(panel.dates, value) for value in cfg["tasks"]["cutoffs"]]
    max_horizon = max(map(int, cfg["tasks"]["demand_horizons_days"]))
    if max(cutoff_indices) + max_horizon > len(panel.dates):
        raise ValueError("a frozen evaluation horizon exceeds the observation calendar")
    encoded, dense, encoder = _fit_metadata(panel, roles["development"])
    pseudo = _pseudo_targets(
        roles["development"],
        panel.item_ids,
        cfg["roles"]["salt"] + "|pseudo-target",
        int(cfg["roles"]["pseudo_target_count_per_cutoff"]),
    )
    train_count = int(round(len(pseudo) * float(cfg["methods"]["component_specific_policy"]["pseudo_target_split"][0])))
    validation_count = int(round(len(pseudo) * float(cfg["methods"]["component_specific_policy"]["pseudo_target_split"][1])))
    train_targets = pseudo[:train_count]
    validation_targets = pseudo[train_count : train_count + validation_count]
    diagnostic_targets = pseudo[train_count + validation_count :]
    train_tasks = _build_policy_tasks(train_targets, roles["development"], encoded, panel.quantities, cutoff_indices, cfg)
    validation_tasks = _build_policy_tasks(validation_targets, roles["development"], encoded, panel.quantities, cutoff_indices, cfg)
    policies = {"component_specific_transfer": [], "matched_shared_relation": []}
    search_rows = []
    for seed in cfg["methods"]["seeds"]:
        for method, shared in (("component_specific_transfer", False), ("matched_shared_relation", True)):
            policy = FactorizedTransferPolicy(
                int(seed),
                candidate_count=int(cfg["methods"]["component_specific_policy"]["random_candidate_count"]),
                shared=shared,
            ).fit(train_tasks, validation_tasks)
            policies[method].append(policy)
            selected = next(row for row in policy.search_ if row["selected"])
            search_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "candidate": selected["candidate"],
                    "training_objective": selected["training_objective"],
                    "validation_objective": selected.get("validation_objective"),
                    "occurrence_logit_shift": policy.occurrence_logit_shift_,
                }
            )
    pd.DataFrame(search_rows).to_csv(output / "policy_selection.csv", index=False)
    global_models = _fit_global_models(dense, panel.quantities, roles["development"], cutoff_indices, cfg)
    zig_models = _fit_zig_models(dense, panel.quantities, roles["development"], cutoff_indices, cfg)

    rows, seed_rows = [], []
    task_cfg = cfg["tasks"]
    for target in roles["confirmation"]:
        for cutoff_label, cutoff in zip(task_cfg["cutoffs"], cutoff_indices, strict=True):
            eligible = _eligible_donors(
                panel.quantities,
                roles["development"],
                cutoff,
                int(task_cfg["required_history_days"]),
                int(cfg["eligibility"]["donor_minimum_positive_history_days"]),
            )
            donors, similarity = _nearest(
                encoded, int(target), eligible, int(task_cfg["donors_per_target"])
            )
            for horizon in task_cfg["demand_horizons_days"]:
                for ratio in task_cfg["shortage_holding_ratios"]:
                    task = _make_task(
                        int(target), donors, similarity, panel.quantities, cutoff, int(horizon),
                        float(ratio), int(task_cfg["required_history_days"]),
                        int(task_cfg["donor_empirical_atoms"]),
                    )
                    distributions: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
                    k = len(donors)
                    complete_weights = {
                        "single_donor": np.eye(k)[0],
                        "complete_uniform": np.full(k, 1.0 / k),
                        "complete_similarity": (
                            similarity / similarity.sum()
                            if similarity.sum() > 0
                            else np.full(k, 1.0 / k)
                        ),
                    }
                    for method, weights in complete_weights.items():
                        values, masses, diagnostic = complete_mixture_distribution(task, weights)
                        distributions[method] = (values, masses, diagnostic["event_probability"])
                    for method in ("matched_shared_relation", "component_specific_transfer"):
                        distributions[method] = _ensemble_distribution(task, policies[method])
                        for seed, policy in zip(cfg["methods"]["seeds"], policies[method], strict=True):
                            values, masses, diagnostic = factorized_distribution(
                                task, policy.parameters_, policy.occurrence_logit_shift_, shared=policy.shared
                            )
                            action = optimal_feasible_level(values, masses, 1.0, float(ratio))[0]
                            cost, service, pinball = _score(action, task.actual_demand, float(ratio))
                            seed_rows.append(
                                {"target_id": int(panel.item_ids[target]), "cutoff": cutoff_label,
                                 "horizon": horizon, "cost_ratio": ratio, "method": method,
                                 "policy_seed": seed, "actual": task.actual_demand, "action": action,
                                 "cost": cost, "service": service, "pinball": pinball,
                                 "event_probability": diagnostic["event_probability"]}
                            )
                    tau = round(float(ratio / (1.0 + ratio)), 7)
                    quantile = max(float(global_models[(str(cutoff_label), int(horizon), tau)].predict(dense[target][None])[0]), 0.0)
                    distributions["rolling_global_quantile"] = (np.asarray([quantile]), np.asarray([1.0]), np.nan)
                    zig_values, zig_masses, zig_event = _zig_distribution(
                        zig_models[(str(cutoff_label), int(horizon))], dense[target],
                        _seed(panel.item_ids[target], cutoff_label, horizon, ratio),
                    )
                    distributions["adapted_zig_mc_catboost"] = (zig_values, zig_masses, zig_event)
                    for method, (values, masses, event_probability) in distributions.items():
                        action = quantile if method == "rolling_global_quantile" else optimal_feasible_level(values, masses, 1.0, float(ratio))[0]
                        cost, service, pinball = _score(action, task.actual_demand, float(ratio))
                        rows.append(
                            {"target_id": int(panel.item_ids[target]), "cutoff": cutoff_label,
                             "horizon": horizon, "cost_ratio": ratio, "method": method,
                             "actual": task.actual_demand, "action": action, "cost": cost,
                             "service": service, "pinball": pinball,
                             "event_probability": event_probability,
                             "brier": (event_probability - float(task.actual_demand > 0)) ** 2
                             if np.isfinite(event_probability) else np.nan}
                        )

    result = pd.DataFrame(rows)
    keys = ["target_id", "cutoff", "horizon", "cost_ratio", "method"]
    if result.duplicated(keys).any():
        raise ValueError("duplicate evaluation keys")
    expected = len(roles["confirmation"]) * len(task_cfg["cutoffs"]) * len(task_cfg["demand_horizons_days"]) * len(task_cfg["shortage_holding_ratios"])
    counts = result.groupby("method").size()
    if not (counts == expected).all():
        raise ValueError(f"missing method rows: {counts.to_dict()}")
    result.to_parquet(output / "row_level_results.parquet", index=False)
    pd.DataFrame(seed_rows).to_parquet(output / "policy_seed_rows.parquet", index=False)
    summary = result.groupby("method", as_index=False).agg(
        cost=("cost", "mean"), service=("service", "mean"), pinball=("pinball", "mean"),
        brier=("brier", "mean"), zero_action_rate=("action", lambda values: float(np.mean(values <= 0))),
        mean_action=("action", "mean"), tail_cost=("cost", lambda values: float(values[values >= values.quantile(0.9)].mean())),
    )
    intervals = _bootstrap(
        result,
        int(cfg["metrics"]["bootstrap_draws"]),
        int(cfg["metrics"]["bootstrap_seed"]),
    )
    summary.merge(intervals, on="method").to_csv(output / "summary.csv", index=False)
    manifest = {
        "status": "completed",
        "protocol_commit": "04cf0e90fdb65d154fa73570851663e8581992b8",
        "config_sha256": _sha256(config_path),
        "input_hashes": {
            "items.csv": _sha256(ROOT / "data/raw/favorita/extracted_v1/csv/items.csv"),
            "stores.csv": _sha256(ROOT / "data/raw/favorita/extracted_v1/csv/stores.csv"),
            "train.csv": _sha256(ROOT / "data/raw/favorita/extracted_v1/csv/train.csv"),
        },
        "role_counts": {name: int(len(values)) for name, values in roles.items()},
        "pseudo_target_counts": {"training": int(len(train_targets)), "validation": int(len(validation_targets)), "held_out_diagnostic": int(len(diagnostic_targets))},
        "encoder_categories": [int(len(values)) for values in encoder.categories_],
        "matched_rows_per_method": {str(name): int(value) for name, value in counts.items()},
        "elapsed_seconds": time.time() - started,
        "target_pre_cutoff_demand_used": False,
        "confirmation_used_for_tuning": False,
        "interpretation": "same-domain training with frozen isolated products; observed aggregate sales, not verified launches or latent demand",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.merge(intervals, on="method").to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
