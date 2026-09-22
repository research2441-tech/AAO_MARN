from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix
from torch.utils.data import DataLoader
from train import ManifestDataset
from model import AAOMARN
from utils import load_config,device,write_json,seed_everything

def ece(y,p,bins=15):
    edges=np.linspace(0,1,bins+1); val=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>lo)&(p<=hi)
        if m.any(): val+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(val)

def compute(y,p,th,bins=15):
    pred=(p>=th).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    specificity=tn/max(tn+fp,1)
    return {'n':int(len(y)),'support_healthy':int((y==0).sum()),'support_diseased':int((y==1).sum()),'threshold':float(th),
            'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
            'recall':recall_score(y,pred,zero_division=0),'specificity':specificity,'f1':f1_score(y,pred,zero_division=0),
            'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'nll':log_loss(y,np.c_[1-p,p],labels=[0,1]),'ece':ece(y,p,bins),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def bootstrap(y,p,th,resamples=2000,cl=.95,seed=2026):
    rng=np.random.default_rng(seed); keys=['accuracy','precision','recall','specificity','f1','roc_auc','pr_auc']; vals={k:[] for k in keys}
    idx0=np.flatnonzero(y==0); idx1=np.flatnonzero(y==1)
    for _ in range(resamples):
        idx=np.r_[rng.choice(idx0,len(idx0),True),rng.choice(idx1,len(idx1),True)]; m=compute(y[idx],p[idx],th)
        for k in keys: vals[k].append(m[k])
    a=(1-cl)/2; return {k:{'low':float(np.quantile(v,a)),'high':float(np.quantile(v,1-a))} for k,v in vals.items()}

def evaluate_checkpoint(cfg,manifest,checkpoint,seed=17,prefix='locked_test'):
    seed_everything(seed); dev=device(); ck=torch.load(checkpoint,map_location='cpu'); kw=ck['kwargs']; m=AAOMARN(**kw).to(dev); m.load_state_dict(ck['state']); m.eval()
    test=manifest[manifest.split=='test']; dl=DataLoader(ManifestDataset(test,cfg,'full',False),batch_size=cfg['training']['batch_size'],shuffle=False)
    rows=[]
    with torch.no_grad():
        for x,y,paths in dl:
            p=torch.sigmoid(m(x.to(dev))).cpu().numpy()
            for a,b,c in zip(paths,y.numpy(),p): rows.append({'path':a,'y_true':int(b),'probability':float(c),'prediction':int(c>=ck['threshold'])})
    pred=pd.DataFrame(rows); pred.to_csv(f'{prefix}_predictions.csv',index=False)
    y=pred.y_true.values; p=pred.probability.values; met=compute(y,p,ck['threshold'],cfg['evaluation']['calibration_bins']); met['confidence_intervals']=bootstrap(y,p,ck['threshold'],cfg['evaluation']['bootstrap_resamples'],cfg['evaluation']['confidence_level'])
    write_json('authoritative_metrics.json',met); return met

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--manifest',default='manifest.csv'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--seed',type=int,default=17); a=ap.parse_args();
    print(evaluate_checkpoint(load_config(a.config),pd.read_csv(a.manifest),a.checkpoint,a.seed))
