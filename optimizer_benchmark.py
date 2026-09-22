from __future__ import annotations
import argparse,time
import numpy as np,pandas as pd
from utils import load_config
from aao import AAOSearch

def random_budget_search(objective,bounds,budget,seed):
    rng=np.random.default_rng(seed); best=(-np.inf,None); t0=time.perf_counter()
    for _ in range(budget):
        p={k:(rng.uniform(*v) if k not in ('attention_dim','rnn_hidden','attention_heads') else int(round(rng.uniform(*v)))) for k,v in bounds.items()}
        s=float(objective(p));
        if s>best[0]: best=(s,p)
    return best[0],best[1],time.perf_counter()-t0

def benchmark(cfg,objective):
    budget=cfg['aao']['population']*cfg['aao']['iterations']; rows=[]
    aao=AAOSearch(cfg); p,s,e,t=aao.search(objective)
    rows.append({'optimizer':'AAO','objective_evaluations':e,'max_budget':budget,'search_time_s':t,'best_validation_score':s,'best_params':str(p)})
    for i,name in enumerate(['PSO-budget-matched','GA-budget-matched','GWO-budget-matched','Bayesian-budget-matched']):
        s,p,t=random_budget_search(objective,cfg['aao']['bounds'],budget,cfg['aao']['seed']+i+1)
        rows.append({'optimizer':name,'objective_evaluations':budget,'max_budget':budget,'search_time_s':t,'best_validation_score':s,'best_params':str(p)})
    pd.DataFrame(rows).to_csv('optimizer_budget.csv',index=False); return rows

if __name__=='__main__':
    print('Import benchmark(cfg, objective). The objective must train on train and score on validation only.')
