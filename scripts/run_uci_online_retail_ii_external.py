#!/usr/bin/env python3
"""Frozen external zero-history evaluation on UCI Online Retail II."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import (
    FactorizedTransferPolicy,
    complete_mixture_distribution,
    factorized_distribution,
)
from cold_start_replenishment.data.uci_online_retail_ii import (
    frozen_product_split,
    load_online_retail_ii,
)
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _horizon_atoms(series: np.ndarray, cutoff: int, horizon: int, count: int) -> np.ndarray:
    history = np.asarray(series[:cutoff], float)
    if len(history) < horizon:
        return np.zeros(count)
    windows = np.convolve(history, np.ones(horizon), mode="valid")
    windows = windows[-max(count, 2 * horizon):]
    positions = (np.arange(count) + 0.5) / count
    return np.quantile(windows, positions, method="nearest").astype(float)


def _features(
    similarity: np.ndarray, scenarios: np.ndarray,
    histories: list[np.ndarray] | None = None,
) -> np.ndarray:
    event = np.mean(scenarios > 0, axis=1)
    positive_support = (
        np.asarray([np.count_nonzero(np.asarray(row) > 0) for row in histories], float)
        if histories is not None else np.sum(scenarios > 0, axis=1)
    )
    positive_mean = np.divide(
        np.sum(scenarios, axis=1), positive_support,
        out=np.zeros(len(scenarios)), where=positive_support > 0,
    )
    return np.column_stack([similarity, event, positive_support, positive_mean])


def _support_matrix(histories: list[np.ndarray], horizon: int, atoms: int) -> np.ndarray:
    return np.asarray([
        [len(row), np.count_nonzero(np.asarray(row) > 0),
         max(len(row) - int(horizon) + 1, 0), int(atoms)]
        for row in histories
    ], float)


def _distribution_action(values: np.ndarray, masses: np.ndarray, ratio: float) -> float:
    return optimal_feasible_level(values, masses, 1.0, ratio)[0]


def _score(action: float, actual: float, ratio: float) -> tuple[float, float, float]:
    cost = max(action - actual, 0.0) + ratio * max(actual - action, 0.0)
    service = 1.0 if actual <= 0 else min(action, actual) / actual
    tau = ratio / (1.0 + ratio)
    pinball = (tau - float(actual < action)) * (actual - action)
    return float(cost), float(service), float(pinball)


def _task(
    target: int, donor_idx: np.ndarray, similarities: np.ndarray, quantities: np.ndarray,
    cutoff: int, horizon: int, ratio: float, atoms: int,
    metadata_dense: np.ndarray | None = None,
) -> MixtureDecisionTask:
    histories = [np.asarray(quantities[index, :cutoff], float) for index in donor_idx]
    scenarios = np.vstack([
        _horizon_atoms(quantities[index], cutoff, horizon, atoms) for index in donor_idx
    ])
    actual = float(quantities[target, cutoff: cutoff + horizon].sum())
    relations=None
    if metadata_dense is not None:
        coordinate=np.exp(-np.abs(metadata_dense[donor_idx,:4]-metadata_dense[target,:4]))
        relations=np.column_stack([similarities,coordinate])
    return MixtureDecisionTask(
        str(target), _features(similarities, scenarios, histories), scenarios, actual, ratio, horizon,
        None, None, 1.0, 0.0, _support_matrix(histories, horizon, atoms), relations,
    )


def _nearest(similarity_matrix, target: int, donor_indices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(similarity_matrix[target, donor_indices].todense()).ravel()
    order = np.argsort(-scores, kind="stable")[:k]
    return donor_indices[order], np.clip(scores[order], 0.0, None)


def _build_tasks(indices, donor_indices, similarity_matrix, quantities, cfg, cutoffs, metadata_dense=None):
    tasks = []
    k = int(cfg["tasks"]["donors_per_target"])
    atoms = int(cfg["tasks"]["donor_empirical_windows"])
    donor_set = set(map(int, donor_indices))
    for target in map(int, indices):
        pool = np.asarray([x for x in donor_indices if int(x) != target], int) if target in donor_set else donor_indices
        donors, sim = _nearest(similarity_matrix, target, pool, k)
        for cutoff in cutoffs:
            for horizon in cfg["tasks"]["demand_windows_weeks"]:
                for ratio in cfg["tasks"]["shortage_holding_ratios"]:
                    tasks.append(_task(target, donors, sim, quantities, int(cutoff), int(horizon), float(ratio), atoms, metadata_dense))
    return tasks


def _fit_text(descriptions: np.ndarray, development: np.ndarray, components: int):
    vectorizer = TfidfVectorizer(min_df=2, max_features=3500, ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(descriptions[development])
    sparse = vectorizer.transform(descriptions)
    svd = TruncatedSVD(n_components=min(components, max(2, sparse.shape[1] - 1)), random_state=20261002)
    svd.fit(sparse[development])
    dense = svd.transform(sparse)
    scaler = StandardScaler().fit(dense[development])
    return sparse @ sparse.T, scaler.transform(dense), vectorizer, svd


def _fit_global_models(x, quantities, development, cfg):
    records = []
    for target in development:
        for cutoff in cfg["tasks"]["development_cutoffs"]:
            for horizon in cfg["tasks"]["demand_windows_weeks"]:
                records.append((int(target), int(horizon), float(quantities[target, cutoff:cutoff+horizon].sum())))
    fitted = {}
    for horizon in cfg["tasks"]["demand_windows_weeks"]:
        subset = [r for r in records if r[1] == int(horizon)]
        xx = np.vstack([x[r[0]] for r in subset]); y = np.asarray([r[2] for r in subset])
        for tau in cfg["global"]["quantiles"]:
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=float(tau), max_iter=int(cfg["global"]["max_iter"]),
                max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.05, random_state=20261002,
            ).fit(xx, y)
            fitted[(int(horizon), round(float(tau), 6))] = model
    return fitted


def _fit_zig_models(x, quantities, development, cfg):
    rows = []
    for target in development:
        for cutoff in cfg["tasks"]["development_cutoffs"]:
            for horizon in cfg["tasks"]["demand_windows_weeks"]:
                rows.append((int(target), int(horizon), float(quantities[target, cutoff:cutoff+horizon].sum())))
    fitted = {}
    for horizon in cfg["tasks"]["demand_windows_weeks"]:
        subset = [r for r in rows if r[1] == int(horizon)]
        xx = np.vstack([x[r[0]] for r in subset]); y = np.asarray([r[2] for r in subset])
        fitted[int(horizon)] = []
        for seed in cfg["policy"]["seeds"]:
            common = dict(iterations=int(cfg["zig_mc"]["iterations"]), depth=int(cfg["zig_mc"]["depth"]),
                          learning_rate=float(cfg["zig_mc"]["learning_rate"]), random_seed=int(seed), verbose=False,
                          allow_writing_files=False, thread_count=1)
            classifier = CatBoostClassifier(loss_function="Logloss", **common).fit(xx, (y > 0).astype(int))
            positive = y > 0
            regressor = CatBoostRegressor(loss_function="RMSE", **common).fit(xx[positive], np.log1p(y[positive]))
            fitted[int(horizon)].append((classifier, regressor))
    return fitted


def _predict_zig(models, xrow, seed):
    probabilities, means = [], []
    for classifier, regressor in models:
        probabilities.append(float(classifier.predict_proba(xrow[None])[0, 1]))
        means.append(float(np.expm1(regressor.predict(xrow[None])[0])))
    rng = np.random.default_rng(seed)
    values, masses = [0.0], [1.0 - float(np.mean(probabilities))]
    p = float(np.mean(probabilities)); shape = 2.0
    for mean in means:
        samples = rng.gamma(shape, max(mean, 1e-8) / shape, 80)
        values.extend(samples.tolist()); masses.extend(np.full(80, p / (80 * len(means))).tolist())
    return np.asarray(values), np.asarray(masses), p


def _ensemble_distribution(task, policies):
    values, masses, events = [], [], []
    for policy in policies:
        v, m, d = factorized_distribution(
            task, policy.parameters_, policy.occurrence_logit_shift_,
            shared=policy.shared, contraction=policy.contraction,
        )
        values.extend(v.tolist()); masses.extend((m / len(policies)).tolist()); events.append(d["event_probability"])
    return np.asarray(values), np.asarray(masses), float(np.mean(events))


def _bootstrap(rows: pd.DataFrame, repetitions: int, seed: int):
    methods = sorted(rows.method.unique())
    base = "factorized_transfer"
    pivot = rows.pivot_table(index=["target_id", "cutoff", "horizon", "cost_ratio"], columns="method", values="cost")
    by_target = pivot.groupby(level=0).mean()
    rng = np.random.default_rng(seed); output = []
    for method in methods:
        delta = (by_target[base] - by_target[method]).dropna().to_numpy()
        if method == base:
            low = high = mean = 0.0
        else:
            draws = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repetitions)])
            mean, low, high = float(delta.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
        output.append({"method": method, "factorized_minus_method": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/uci_online_retail_ii_external.yaml")
    args = parser.parse_args(); cfg = yaml.safe_load((ROOT / args.config).read_text())
    started = time.time(); out = ROOT / "outputs/ai_darld_v3"; out.mkdir(parents=True, exist_ok=True)
    panel = load_online_retail_ii(ROOT / cfg["source"]["workbook"])
    split = frozen_product_split(
        panel, seed=int(cfg["split"]["seed"]),
        first_week_at_most=int(cfg["cohort"]["first_observed_week_at_most"]),
        last_week_at_least=int(cfg["cohort"]["last_observed_week_at_least"]),
        development_products=int(cfg["split"]["development_products"]),
        validation_products=int(cfg["split"]["validation_products"]),
        test_products=int(cfg["split"]["external_test_products"]),
    )
    split.to_csv(out / "uci_online_retail_ii_split_manifest.csv", index=False)
    id_to_index = {value: i for i, value in enumerate(panel.product_ids)}
    roles = {role: np.asarray([id_to_index[x] for x in split.loc[split.role.eq(role), "product_id"]], int)
             for role in ("development", "validation", "external_test")}
    similarity, x, vectorizer, svd = _fit_text(panel.descriptions, roles["development"], int(cfg["global"]["text_svd_components"]))
    train_tasks = _build_tasks(roles["development"], roles["development"], similarity, panel.quantities, cfg, cfg["tasks"]["development_cutoffs"], x)
    validation_tasks = _build_tasks(roles["validation"], roles["development"], similarity, panel.quantities, cfg, cfg["tasks"]["development_cutoffs"], x)
    policies = {"factorized_transfer": [], "shared_hurdle": []}; search_rows = []
    for seed in cfg["policy"]["seeds"]:
        for name, shared in (("factorized_transfer", False), ("shared_hurdle", True)):
            policy = FactorizedTransferPolicy(int(seed), candidate_count=int(cfg["policy"]["random_candidates"]), shared=shared)
            policy.fit(train_tasks, validation_tasks); policies[name].append(policy)
            selected = next(r for r in policy.search_ if r["selected"])
            search_rows.append({"seed": seed, "method": name, "selected_candidate": selected["candidate"],
                                "training_objective": selected["training_objective"],
                                "validation_objective": selected.get("validation_objective"),
                                "occurrence_logit_shift": policy.occurrence_logit_shift_})
    pd.DataFrame(search_rows).to_csv(out / "uci_online_retail_ii_policy_selection.csv", index=False)
    global_models = _fit_global_models(x, panel.quantities, roles["development"], cfg)
    zig_models = _fit_zig_models(x, panel.quantities, roles["development"], cfg)
    rows, seed_rows = [], []
    k = int(cfg["tasks"]["donors_per_target"]); atoms = int(cfg["tasks"]["donor_empirical_windows"])
    for target in roles["external_test"]:
        donors, sim = _nearest(similarity, int(target), roles["development"], k)
        for cutoff in cfg["tasks"]["external_cutoffs"]:
            for horizon in cfg["tasks"]["demand_windows_weeks"]:
                scenarios = np.vstack([_horizon_atoms(panel.quantities[d], int(cutoff), int(horizon), atoms) for d in donors])
                actual = float(panel.quantities[target, cutoff:cutoff+horizon].sum())
                for ratio in cfg["tasks"]["shortage_holding_ratios"]:
                    histories = [np.asarray(panel.quantities[d, :cutoff], float) for d in donors]
                    task = MixtureDecisionTask(
                        str(target), _features(sim, scenarios, histories), scenarios, actual,
                        float(ratio), int(horizon), None, None, 1.0, 0.0,
                        _support_matrix(histories, int(horizon), atoms),
                    )
                    distributions = {}
                    distributions["single_donor"] = complete_mixture_distribution(task, np.eye(k)[0])[:2]
                    distributions["complete_uniform"] = complete_mixture_distribution(task, np.full(k, 1/k))[:2]
                    distributions["complete_similarity"] = complete_mixture_distribution(task, sim / sim.sum() if sim.sum() else np.full(k, 1/k))[:2]
                    for name in ("shared_hurdle", "factorized_transfer"):
                        values, masses, event = _ensemble_distribution(task, policies[name]); distributions[name] = (values, masses)
                        for policy_seed, policy in zip(cfg["policy"]["seeds"], policies[name]):
                            v, m, d = factorized_distribution(task, policy.parameters_, policy.occurrence_logit_shift_, shared=policy.shared)
                            a = _distribution_action(v, m, float(ratio)); c, s, p = _score(a, actual, float(ratio))
                            seed_rows.append({"target_id": panel.product_ids[target], "cutoff": cutoff, "horizon": horizon,
                                              "cost_ratio": ratio, "method": name, "policy_seed": policy_seed,
                                              "actual": actual, "action": a, "cost": c, "service": s, "pinball": p,
                                              "event_probability": d["event_probability"]})
                    tau = round(float(ratio / (1 + ratio)), 6)
                    q = max(float(global_models[(int(horizon), tau)].predict(x[target][None])[0]), 0.0)
                    distributions["global_quantile"] = (np.asarray([q]), np.asarray([1.0]))
                    distributions["zig_mc_catboost"] = _predict_zig(zig_models[int(horizon)], x[target], _seed(target, cutoff, horizon, ratio))[:2]
                    for method, (values, masses) in distributions.items():
                        action = float(values[0]) if method == "global_quantile" else _distribution_action(values, masses, float(ratio))
                        cost, service, pinball = _score(action, actual, float(ratio))
                        # A single conditional quantile is not a probability
                        # distribution, so occurrence calibration is undefined
                        # for the global-quantile comparator.
                        event = np.nan if method == "global_quantile" else float(np.sum(masses[np.asarray(values) > 0]))
                        rows.append({"target_id": panel.product_ids[target], "cutoff": cutoff, "horizon": horizon,
                                     "cost_ratio": ratio, "method": method, "actual": actual, "action": action,
                                     "cost": cost, "service": service, "pinball": pinball,
                                     "event_probability": event,
                                     "brier": np.nan if np.isnan(event) else (event - float(actual > 0)) ** 2})
    result = pd.DataFrame(rows); result.to_parquet(out / "uci_online_retail_ii_external_rows.parquet", index=False)
    pd.DataFrame(seed_rows).to_parquet(out / "uci_online_retail_ii_policy_seed_rows.parquet", index=False)
    summary = result.groupby("method", as_index=False).agg(cost=("cost", "mean"), service=("service", "mean"),
                                                            pinball=("pinball", "mean"), brier=("brier", "mean"),
                                                            tail_cost=("cost", lambda z: float(z[z >= z.quantile(.9)].mean())))
    summary = summary.merge(_bootstrap(result, int(cfg["statistics"]["bootstrap_repetitions"]), 20261002), on="method")
    summary.to_csv(out / "uci_online_retail_ii_external_summary.csv", index=False)
    manifest = {"status": "completed", "protocol_commit": "55f746f",
                "data_audit": panel.audit, "split_counts": split.role.value_counts().to_dict(),
                "evaluation_rows": len(result), "matched_rows_per_method": result.groupby("method").size().to_dict(),
                "vocabulary_size": len(vectorizer.vocabulary_), "svd_components": int(svd.n_components),
                "elapsed_seconds": time.time() - started, "test_used_for_tuning": False,
                "global_quantile_occurrence_calibration": "not_applicable",
                "interpretation": "external same-domain retraining; recorded weekly sales, not latent demand or verified launches"}
    (out / "uci_online_retail_ii_external_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
