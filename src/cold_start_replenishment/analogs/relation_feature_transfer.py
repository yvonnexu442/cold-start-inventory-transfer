"""Learn occurrence and magnitude weights from deployable relation features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import _softmax, _zscore
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level


def relation_distribution(task: MixtureDecisionTask, theta: NDArray[np.float64], *, shared: bool=False):
    if task.donor_relations is None:
        raise ValueError("relation features are required")
    relation=np.asarray(task.donor_relations,float)
    if relation.ndim != 2 or relation.shape[0] != len(task.donor_scenarios):
        raise ValueError("relations must be KxR")
    r=relation.shape[1]; theta=np.asarray(theta,float)
    if theta.shape != (2*r+4,):
        raise ValueError("parameter dimension must be 2R+4")
    rz=_zscore(relation)
    event=np.mean(task.donor_scenarios>0,axis=1)
    positive=np.asarray([x[x>0].mean() if np.any(x>0) else 0 for x in task.donor_scenarios])
    support=(task.donor_support[:,1] if task.donor_support is not None else task.donor_features[:,2])
    oz=np.column_stack([rz,_zscore(np.column_stack([event,np.log1p(support)]))])
    mz=np.column_stack([rz,_zscore(np.column_stack([positive,np.log1p(support)]))])
    wo=_softmax(oz@np.r_[theta[:r],theta[2*r:2*r+2]])
    wm=_softmax(mz@np.r_[theta[r:2*r],theta[2*r+2:2*r+4]])
    if shared: wm=wo.copy()
    p=float(np.clip(wo@event,0.0,1.0)); values=[0.0]; masses=[max(0.0,1-p)]
    active=[j for j,x in enumerate(task.donor_scenarios) if np.any(x>0)]
    if active and p>0:
        aw=wm[active]/wm[active].sum()
        for j,w in zip(active,aw,strict=True):
            x=task.donor_scenarios[j][task.donor_scenarios[j]>0]
            values.extend(x.tolist()); masses.extend(np.full(len(x),p*w/len(x)).tolist())
    masses=np.asarray(masses,float); masses=masses/masses.sum()
    return np.asarray(values),masses,{"event_probability":p,"occurrence_weights":wo,"magnitude_weights":wm}


def relation_loss(task,theta,*,shared=False,scale=1.0):
    v,m,d=relation_distribution(task,theta,shared=shared)
    q=optimal_feasible_level(v,m,task.holding_cost,task.holding_cost*task.cost_ratio,
        capacity=task.capacity,minimum_order_quantity=task.minimum_order_quantity,
        fixed_order_cost=task.fixed_order_cost)[0]
    y=task.actual_demand
    cost=task.holding_cost*max(q-y,0)+task.holding_cost*task.cost_ratio*max(y-q,0)+(task.fixed_order_cost if q>0 else 0)
    brier=(d["event_probability"]-float(y>0))**2
    pm=m[v>0].sum(); mean=float(np.sum(v[v>0]*m[v>0])/pm) if pm>0 else 0
    size=abs(mean-y)/max(scale,1.0) if y>0 else 0
    return float(cost/max(scale,1.0)+.1*(brier+size))


@dataclass
class RelationFeaturePolicy:
    random_seed:int
    candidate_count:int=96
    shared:bool=False
    objective_scale:str="cohort_median"

    def fit(self,training:list[MixtureDecisionTask],validation:list[MixtureDecisionTask]):
        r=training[0].donor_relations.shape[1]
        rng=np.random.default_rng(self.random_seed)
        candidates=np.vstack([np.zeros((1,2*r+4)),rng.normal(0,.65,(self.candidate_count,2*r+4))])
        positive=[t.actual_demand for t in training if t.actual_demand>0]
        scale=float(np.median(positive)) if self.objective_scale=="cohort_median" and positive else 1.0
        records=[]
        for i,x in enumerate(candidates):
            tr=float(np.mean([relation_loss(t,x,shared=self.shared,scale=scale) for t in training]))
            records.append({"candidate":i,"training_objective":tr})
        shortlist=sorted(records,key=lambda z:z["training_objective"])[:12]
        for row in shortlist:
            x=candidates[row["candidate"]]
            row["validation_objective"]=float(np.mean([
                relation_loss(t,x,shared=self.shared,scale=scale) for t in validation]))
        best=min(shortlist,key=lambda z:z["validation_objective"])["candidate"]
        for row in records: row["selected"]=row["candidate"]==best
        self.parameters_=candidates[best].copy(); self.search_=records; self.scale_=scale
        return self
