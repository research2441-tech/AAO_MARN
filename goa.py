from __future__ import annotations
import argparse, math, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from utils import load_config, seed_everything

def position_to_mask(z,min_channels=32,max_channels=256,threshold=0.5):
    p=1/(1+np.exp(-np.clip(z,-30,30))); m=p>=threshold; k=int(m.sum())
    order=np.argsort(-p)
    if k<min_channels:
        m[:]=False; m[order[:min_channels]]=True
    elif k>max_channels:
        m[:]=False; m[order[:max_channels]]=True
    return m,p

def social_function(r,f=0.5,l=1.5): return f*np.exp(-r/l)-np.exp(-r)

def optimize(Xtr,ytr,Xv,yv,cfg,history_path='goa_history.csv'):
    g=cfg['goa']; seed_everything(int(g['seed'])); rng=np.random.default_rng(int(g['seed']))
    P,G,D=int(g['population']),int(g['generations']),Xtr.shape[1]
    pos=rng.normal(0,1,(P,D)); best=None; hist=[]; t0=time.perf_counter()
    for gen in range(G):
        c=g['c_max']-(g['c_max']-g['c_min'])*gen/max(G-1,1)
        fits=[]; masks=[]
        for i in range(P):
            mask,_=position_to_mask(pos[i],g['min_channels'],g['max_channels'])
            clf=LogisticRegression(max_iter=300,class_weight='balanced',solver='liblinear',random_state=int(g['seed']))
            clf.fit(Xtr[:,mask],ytr); pred=clf.predict(Xv[:,mask]); ba=balanced_accuracy_score(yv,pred)
            fit=g['alpha']*ba+g['beta']*(1-mask.sum()/D); fits.append(fit); masks.append(mask)
            if best is None or fit>best[0]: best=(fit,pos[i].copy(),mask.copy(),ba)
        target=best[1]
        new=np.zeros_like(pos)
        for i in range(P):
            s=np.zeros(D)
            for j in range(P):
                if i==j: continue
                diff=pos[j]-pos[i]; dist=np.linalg.norm(diff)+1e-12
                s+=social_function(dist)*(diff/dist)
            new[i]=c*s+target
        pos=new
        hist.append({'generation':gen+1,'best_fitness':best[0],'best_balanced_accuracy':best[3],
                     'selected_channels':int(best[2].sum()),'c':c,'elapsed_s':time.perf_counter()-t0})
    pd.DataFrame(hist).to_csv(history_path,index=False)
    return np.flatnonzero(best[2]),best[0]

if __name__=='__main__':
    print('GOA module ready. Use optimize(X_train, y_train, X_val, y_val, cfg).')
