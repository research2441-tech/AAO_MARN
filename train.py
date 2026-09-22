from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as TF
from sklearn.metrics import balanced_accuracy_score
from preprocessing import apply_pipeline
from model import AAOMARN
from utils import load_config,seed_everything,device,write_json

class ManifestDataset(Dataset):
    def __init__(self,df,cfg,variant='full',augment=False): self.df=df.reset_index(drop=True); self.cfg=cfg; self.variant=variant; self.augment=augment
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=cv2.cvtColor(cv2.imread(r.path),cv2.COLOR_BGR2RGB); im=apply_pipeline(im,self.cfg,self.variant,seed=i)
        im=cv2.resize(im,(self.cfg['image_size'],self.cfg['image_size']))
        x=torch.from_numpy(im).permute(2,0,1).float()/255.; x=TF.normalize(x,[0.485,0.456,0.406],[0.229,0.224,0.225])
        return x,torch.tensor(float(r.label)),r.path

def best_threshold(y,p):
    cand=np.linspace(0.05,0.95,181); scores=[balanced_accuracy_score(y,p>=t) for t in cand]; return float(cand[int(np.argmax(scores))])

def train_one(cfg,manifest,seed=17,selected_idx=None,variant='full',model_kwargs=None,save_prefix='model'):
    seed_everything(seed); dev=device(); tr=manifest[manifest.split=='train']; va=manifest[manifest.split=='val']
    bs=cfg['training']['batch_size']; nw=cfg['training']['num_workers']
    dltr=DataLoader(ManifestDataset(tr,cfg,variant,True),batch_size=bs,shuffle=True,num_workers=nw)
    dlva=DataLoader(ManifestDataset(va,cfg,variant,False),batch_size=bs,shuffle=False,num_workers=nw)
    kw=dict(attention_dim=cfg['model']['attention_dim'],heads=cfg['model']['attention_heads'],rnn_hidden=cfg['model']['rnn_hidden'],dropout=cfg['model']['dropout'],pretrained=cfg['model']['pretrained'])
    if model_kwargs: kw.update(model_kwargs)
    m=AAOMARN(selected_idx=selected_idx,**kw).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']); lossfn=torch.nn.BCEWithLogitsLoss()
    best=(-1,None); patience=0; hist=[]
    for ep in range(cfg['training']['epochs']):
        m.train(); ls=[]
        for x,y,_ in dltr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); z=m(x); loss=lossfn(z,y); loss.backward(); opt.step(); ls.append(loss.item())
        m.eval(); pp=[]; yy=[]
        with torch.no_grad():
            for x,y,_ in dlva:
                p=torch.sigmoid(m(x.to(dev))).cpu().numpy(); pp.extend(p); yy.extend(y.numpy())
        th=best_threshold(np.asarray(yy),np.asarray(pp)); ba=balanced_accuracy_score(yy,np.asarray(pp)>=th)
        hist.append({'epoch':ep+1,'train_loss':float(np.mean(ls)),'val_balanced_accuracy':ba,'threshold':th})
        if ba>best[0]+cfg['training']['min_delta']:
            best=(ba,{'state':{k:v.detach().cpu() for k,v in m.state_dict().items()},'threshold':th,'kwargs':kw}); patience=0
        else: patience+=1
        if patience>=cfg['training']['patience']: break
    torch.save(best[1],f'{save_prefix}_seed{seed}.pt'); pd.DataFrame(hist).to_csv(f'{save_prefix}_seed{seed}_history.csv',index=False)
    return best[0],best[1]['threshold']

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--manifest',default='manifest.csv'); ap.add_argument('--seed',type=int,default=17); a=ap.parse_args()
    cfg=load_config(a.config); df=pd.read_csv(a.manifest); print(train_one(cfg,df,a.seed))
