from __future__ import annotations
import argparse,pandas as pd,torch
from torch.utils.data import DataLoader
from train import ManifestDataset
from model import AAOMARN
from evaluate import compute
from utils import load_config,device,write_json

def run(cfg,external_manifest,checkpoint):
    dev=device(); ck=torch.load(checkpoint,map_location='cpu'); m=AAOMARN(**ck['kwargs']).to(dev); m.load_state_dict(ck['state']); m.eval()
    dl=DataLoader(ManifestDataset(external_manifest,cfg,'full',False),batch_size=cfg['training']['batch_size'],shuffle=False)
    y=[]; p=[]; paths=[]
    with torch.no_grad():
        for x,t,ps in dl: p.extend(torch.sigmoid(m(x.to(dev))).cpu().numpy()); y.extend(t.numpy()); paths.extend(ps)
    out=pd.DataFrame({'path':paths,'y_true':y,'probability':p}); out['prediction']=(out.probability>=ck['threshold']).astype(int); out.to_csv('external_predictions.csv',index=False)
    met=compute(out.y_true.values,out.probability.values,ck['threshold'],cfg['evaluation']['calibration_bins']); write_json('external_metrics.json',met); return met
if __name__=='__main__':
    print('Provide a labeled external manifest and frozen checkpoint. Do not tune on external labels.')
