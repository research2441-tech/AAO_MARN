from __future__ import annotations
import argparse,pandas as pd
from utils import load_config
from train import train_one

PREPROCESS=['none','clahe','slic','kmeans','slic_kmeans','clahe_slic_kmeans','full']
COMPONENTS={
'full':{},
'no_mha':{'use_mha':False},
'no_rnn':{'use_rnn':False},
'attention_only':{'use_mha':True,'use_rnn':False},
'rnn_only':{'use_mha':False,'use_rnn':True},
'spatial_2d':{'use_mha':False,'use_rnn':False,'use_spatial2d':True},
}

def run(cfg,df):
    rows=[]
    seeds=cfg['training']['seeds']
    for v in PREPROCESS:
        for s in seeds:
            ba,th=train_one(cfg,df,s,variant=v,save_prefix=f'ablation_pre_{v}')
            rows.append({'family':'preprocessing','variant':v,'seed':s,'val_balanced_accuracy':ba,'threshold':th})
    for name,kw in COMPONENTS.items():
        for s in seeds:
            ba,th=train_one(cfg,df,s,variant='full',model_kwargs=kw,save_prefix=f'ablation_component_{name}')
            rows.append({'family':'component','variant':name,'seed':s,'val_balanced_accuracy':ba,'threshold':th})
    out=pd.DataFrame(rows); out[out.family=='preprocessing'].to_csv('preprocessing_ablation.csv',index=False); out[out.family=='component'].to_csv('component_ablation.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--manifest',default='manifest.csv'); a=ap.parse_args(); run(load_config(a.config),pd.read_csv(a.manifest))
