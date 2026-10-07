#!/usr/bin/env python3
"""Frozen evaluation on previously unused UCI products after support correction."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))

from run_uci_online_retail_ii_external import (
    _bootstrap, _build_tasks, _distribution_action, _ensemble_distribution, _features,
    _fit_global_models, _fit_text, _fit_zig_models, _horizon_atoms, _nearest, _predict_zig, _score,
    _support_matrix,
)
from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import FactorizedTransferPolicy, complete_mixture_distribution, factorized_distribution
from cold_start_replenishment.data.uci_online_retail_ii import load_online_retail_ii


_TRAIN_TASKS = None
_VALIDATION_TASKS = None


def _init_policy_worker(train_tasks, validation_tasks):
    global _TRAIN_TASKS, _VALIDATION_TASKS
    _TRAIN_TASKS, _VALIDATION_TASKS = train_tasks, validation_tasks


def _fit_policy_job(args):
    name, shared, residual, seed = args
    policy = FactorizedTransferPolicy(
        int(seed), candidate_count=96, shared=shared, similarity_residual=residual
    ).fit(_TRAIN_TASKS, _VALIDATION_TASKS)
    selected = next(row for row in policy.search_ if row["selected"])
    record = {
        "seed": seed,
        "method": name,
        "selected_candidate": selected["candidate"],
        "training_objective": selected["training_objective"],
        "validation_objective": selected.get("validation_objective"),
        "occurrence_logit_shift": policy.occurrence_logit_shift_,
    }
    return name, policy, record


def _ordered_eligible(panel, base):
    table=panel.product_table
    eligible=table.loc[table.first_week.le(base["cohort"]["first_observed_week_at_most"]) & table.last_week.ge(base["cohort"]["last_observed_week_at_least"])].copy()
    seed=int(base["split"]["seed"])
    eligible["split_key"]=eligible.product_id.map(lambda x:hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    return eligible.sort_values(["split_key","product_id"],kind="stable").reset_index(drop=True)


def main():
    base=yaml.safe_load((ROOT/"configs/uci_online_retail_ii_external.yaml").read_text())
    frozen=yaml.safe_load((ROOT/"configs/mechanism_confirmation_v3.yaml").read_text())["uci_confirmation_v3"]
    started=time.time(); out=ROOT/"outputs/ai_darld_v3"
    panel=load_online_retail_ii(ROOT/base["source"]["workbook"]); ordered=_ordered_eligible(panel,base)
    old=pd.read_csv(out/"uci_online_retail_ii_split_manifest.csv")
    prior=set(old.product_id.astype(str))
    prior_v2=out/"uci_confirmation_v2_split_manifest.csv"
    if prior_v2.exists():
        prior |= set(pd.read_csv(prior_v2).product_id.astype(str))
    start=int(frozen["prior_assigned_products"]); stop=start+int(frozen["confirmation_products"])
    confirmation=ordered.iloc[start:stop].copy(); assert not (set(confirmation.product_id.astype(str)) & prior)
    confirmation.assign(role="confirmation_v3").to_csv(out/"uci_confirmation_v3_split_manifest.csv",index=False)
    id_to_index={x:i for i,x in enumerate(panel.product_ids)}
    development=np.asarray([id_to_index[x] for x in old.loc[old.role.eq("development"),"product_id"]],int)
    validation=np.asarray([id_to_index[x] for x in old.loc[old.role.eq("validation"),"product_id"]],int)
    targets=np.asarray([id_to_index[x] for x in confirmation.product_id],int)
    similarity,x,vectorizer,svd=_fit_text(panel.descriptions,development,int(base["global"]["text_svd_components"]))
    policy_dev=development[:int(frozen["policy_development_products"])]
    train_tasks=_build_tasks(policy_dev,development,similarity,panel.quantities,base,base["tasks"]["development_cutoffs"],x)
    validation_tasks=_build_tasks(validation,development,similarity,panel.quantities,base,base["tasks"]["development_cutoffs"],x)
    specs=(
        ("shared_hurdle",True,False),("factorized_original",False,False),
        ("shared_similarity_residual",True,True),("factorized_similarity_residual",False,True),
    ); policies={name:[] for name,_,_ in specs}; search=[]
    jobs = [
        (name, shared, residual, seed)
        for seed in frozen["policy_seeds"]
        for name, shared, residual in specs
    ]
    with ProcessPoolExecutor(
        max_workers=min(6, len(jobs)),
        initializer=_init_policy_worker,
        initargs=(train_tasks, validation_tasks),
    ) as executor:
        for name, policy, record in executor.map(_fit_policy_job, jobs):
            policies[name].append(policy)
            search.append(record)
    for name in policies:
        policies[name].sort(key=lambda policy: policy.random_seed)
    pd.DataFrame(search).to_csv(out/"uci_confirmation_v3_policy_selection.csv",index=False)
    global_models=_fit_global_models(x,panel.quantities,development,base); zig_models=_fit_zig_models(x,panel.quantities,development,base)
    rows=[]; seed_rows=[]; k=int(base["tasks"]["donors_per_target"]); atoms=int(base["tasks"]["donor_empirical_windows"])
    for target in targets:
        donors,sim=_nearest(similarity,int(target),development,k)
        for cutoff in frozen["cutoffs"]:
            for horizon in frozen["demand_windows_weeks"]:
                scenarios=np.vstack([_horizon_atoms(panel.quantities[d],int(cutoff),int(horizon),atoms) for d in donors])
                actual=float(panel.quantities[target,cutoff:cutoff+horizon].sum())
                for ratio in frozen["shortage_holding_ratios"]:
                    histories=[np.asarray(panel.quantities[d,:cutoff],float) for d in donors]
                    relations=np.column_stack([sim,np.exp(-np.abs(x[donors,:4]-x[target,:4]))])
                    task=MixtureDecisionTask(
                        str(target),_features(sim,scenarios,histories),scenarios,actual,float(ratio),
                        int(horizon),None,None,1.0,0.0,_support_matrix(histories,int(horizon),atoms),relations
                    )
                    distributions={
                        "single_donor":complete_mixture_distribution(task,np.eye(k)[0])[:2],
                        "complete_uniform":complete_mixture_distribution(task,np.full(k,1/k))[:2],
                        "complete_similarity":complete_mixture_distribution(task,sim/sim.sum() if sim.sum() else np.full(k,1/k))[:2],
                    }
                    for name in policies:
                        values=[]; masses=[]
                        for policy in policies[name]:
                            v,m,d=factorized_distribution(task,policy.parameters_,policy.occurrence_logit_shift_,shared=policy.shared,similarity_residual=policy.similarity_residual)
                            values.extend(v); masses.extend(m/len(policies[name]))
                        distributions[name]=(np.asarray(values),np.asarray(masses))
                    tau=round(float(ratio/(1+ratio)),6); q=max(float(global_models[(int(horizon),tau)].predict(x[target][None])[0]),0)
                    distributions["global_quantile"]=(np.asarray([q]),np.asarray([1.0]))
                    distributions["zig_mc_catboost"]=_predict_zig(zig_models[int(horizon)],x[target],int(target)+int(cutoff)*101+int(horizon)*1009)[:2]
                    for method,(values,masses) in distributions.items():
                        action=float(values[0]) if method=="global_quantile" else _distribution_action(values,masses,float(ratio)); cost,service,pinball=_score(action,actual,float(ratio))
                        event=np.nan if method=="global_quantile" else float(masses[np.asarray(values)>0].sum())
                        pmass=masses[np.asarray(values)>0].sum(); pmean=float(np.sum(np.asarray(values)[np.asarray(values)>0]*masses[np.asarray(values)>0])/pmass) if pmass>0 else 0.0
                        rows.append({"target_id":panel.product_ids[target],"cutoff":cutoff,"horizon":horizon,"cost_ratio":ratio,"method":method,
                                     "actual":actual,"action":action,"cost":cost,"service":service,"pinball":pinball,"event_probability":event,
                                     "brier":np.nan if np.isnan(event) else (event-float(actual>0))**2,
                                     "positive_mean_absolute_error":abs(pmean-actual) if actual>0 else np.nan})
    result=pd.DataFrame(rows); result.to_parquet(out/"uci_confirmation_v3_rows.parquet",index=False)
    summary=result.groupby("method",as_index=False).agg(cost=("cost","mean"),service=("service","mean"),pinball=("pinball","mean"),brier=("brier","mean"),
        positive_mae=("positive_mean_absolute_error","mean"),tail_cost=("cost",lambda z:float(z[z>=z.quantile(.9)].mean())))
    comparisons=_bootstrap(result.rename(columns={"method":"method"}).assign(method=result.method.replace({"factorized_similarity_residual":"factorized_transfer"})),2000,20261103)
    comparisons.method=comparisons.method.replace({"factorized_transfer":"factorized_similarity_residual"}); summary=summary.merge(comparisons,on="method",how="left")
    summary.to_csv(out/"uci_confirmation_v3_summary.csv",index=False)
    manifest={"status":"completed","config":"configs/mechanism_confirmation_v3.yaml","confirmation_products":len(targets),"prior_product_overlap":0,
              "methods_frozen_before_confirmation":["single_donor","complete_uniform","complete_similarity","shared_hurdle","factorized_original","shared_similarity_residual","factorized_similarity_residual","global_quantile","zig_mc_catboost"],
              "rows":len(result),"matched_rows_per_method":result.groupby("method").size().to_dict(),"test_used_for_tuning":False,"elapsed_seconds":time.time()-started}
    (out/"uci_confirmation_v3_manifest.json").write_text(json.dumps(manifest,indent=2)); print(summary.to_string(index=False))


if __name__=="__main__": main()
