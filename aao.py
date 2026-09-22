from __future__ import annotations
import math,time,numpy as np,pandas as pd
from utils import seed_everything

class AAOSearch:
    def __init__(self,cfg):
        self.cfg=cfg['aao']; self.rng=np.random.default_rng(int(self.cfg['seed'])); seed_everything(int(self.cfg['seed']))
        self.keys=['learning_rate','weight_decay','dropout','attention_dim','rnn_hidden','attention_heads']
    def decode(self,z):
        out={}
        for i,k in enumerate(self.keys):
            lo,hi=self.cfg['bounds'][k]; v=lo+np.clip(z[i],0,1)*(hi-lo)
            if k in ('attention_dim','rnn_hidden','attention_heads'): v=int(round(v))
            out[k]=v
        # make attention dimension compatible with head count
        h=max(1,int(out['attention_heads'])); d=max(h,int(out['attention_dim'])); d=max(h,(d//h)*h); out['attention_dim']=d
        return out
    def search(self,objective,history_path='aao_history.csv'):
        P,I=int(self.cfg['population']),int(self.cfg['iterations']); pop=self.rng.random((P,len(self.keys)))
        starvation=np.zeros(P,int); best=(-np.inf,None,None); hist=[]; evals=0; noimp=0; t0=time.perf_counter()
        for it in range(I):
            scores=[]
            for i,z in enumerate(pop):
                params=self.decode(z); score=float(objective(params)); evals+=1; scores.append(score)
                if score>best[0]+self.cfg['min_delta']: best=(score,z.copy(),params.copy()); noimp=0
            elite=np.asarray(best[1]); order=np.argsort(scores)[::-1]
            for i in range(P):
                neigh=pop[self.rng.choice(order[:max(2,P//2)])]
                phase=self.rng.uniform(0,2*np.pi); helix=np.sin(phase)*(neigh-pop[i])
                candidate=pop[i]+self.rng.uniform(0.2,0.8)*helix+self.rng.uniform(0,0.3)*(elite-pop[i])
                if self.rng.random()<self.cfg['adaptation_probability'] or starvation[i]>=3:
                    candidate+=self.rng.normal(0,0.08,size=candidate.shape); starvation[i]=0
                pop[i]=np.clip(candidate,0,1)
                starvation[i]=starvation[i]+1 if scores[i]<best[0] else 0
            noimp+=1
            hist.append({'iteration':it+1,'best_score':best[0],'objective_evaluations':evals,
                         'elapsed_s':time.perf_counter()-t0,**{f'best_{k}':v for k,v in best[2].items()}})
            if noimp>=int(self.cfg['patience']): break
        pd.DataFrame(hist).to_csv(history_path,index=False)
        return best[2],best[0],evals,time.perf_counter()-t0
