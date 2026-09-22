from __future__ import annotations
import argparse, time
import pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import timm
from train import ManifestDataset,best_threshold
from utils import load_config,seed_everything,device
from sklearn.metrics import balanced_accuracy_score

def make_model(name):
    if name=='mobilenetv2':
        m=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1); m.classifier[1]=nn.Linear(m.last_channel,1); return m
    if name=='efficientnet_b0':
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier[1]=nn.Linear(m.classifier[1].in_features,1); return m
    if name=='convnext_tiny': return timm.create_model('convnext_tiny',pretrained=True,num_classes=1)
    if name=='vit_b_16': return timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=1)
    if name=='swin_t': return timm.create_model('swin_tiny_patch4_window7_224',pretrained=True,num_classes=1)
    raise ValueError(name)

def run_one(name,cfg,df,seed):
    seed_everything(seed); dev=device(); tr=df[df.split=='train']; va=df[df.split=='val']; bs=cfg['training']['batch_size']
    dltr=DataLoader(ManifestDataset(tr,cfg,'full',True),batch_size=bs,shuffle=True); dlva=DataLoader(ManifestDataset(va,cfg,'full',False),batch_size=bs)
    m=make_model(name).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']); lossf=nn.BCEWithLogitsLoss(); best=-1; t0=time.perf_counter()
    for _ in range(cfg['training']['epochs']):
        m.train()
        for x,y,_ in dltr:
            opt.zero_grad(); loss=lossf(m(x.to(dev)).view(-1),y.to(dev)); loss.backward(); opt.step()
        m.eval(); p=[]; yv=[]
        with torch.no_grad():
            for x,y,_ in dlva: p.extend(torch.sigmoid(m(x.to(dev)).view(-1)).cpu().numpy()); yv.extend(y.numpy())
        th=best_threshold(yv,p); ba=balanced_accuracy_score(yv,[q>=th for q in p]); best=max(best,ba)
    return {'model':name,'seed':seed,'val_balanced_accuracy':best,'train_wall_time_s':time.perf_counter()-t0}

def run(cfg,df):
    rows=[run_one(n,cfg,df,s) for n in cfg['baseline_models'] for s in cfg['training']['seeds']]; pd.DataFrame(rows).to_csv('baseline_results.csv',index=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--manifest',default='manifest.csv'); a=ap.parse_args(); run(load_config(a.config),pd.read_csv(a.manifest))
