from __future__ import annotations
import argparse,time,psutil
import numpy as np,pandas as pd,torch
from model import AAOMARN
from utils import load_config,device

def measure(cfg,selected_idx=None,warmup=20,runs=100):
    dev=device(); m=AAOMARN(selected_idx=selected_idx,attention_dim=cfg['model']['attention_dim'],heads=cfg['model']['attention_heads'],rnn_hidden=cfg['model']['rnn_hidden'],dropout=cfg['model']['dropout'],pretrained=cfg['model']['pretrained']).to(dev).eval()
    x=torch.randn(1,3,cfg['image_size'],cfg['image_size'],device=dev); total=sum(p.numel() for p in m.parameters()); trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup): m(x)
        if dev.type=='cuda': torch.cuda.synchronize()
        ts=[]
        for _ in range(runs):
            t=time.perf_counter(); m(x)
            if dev.type=='cuda': torch.cuda.synchronize()
            ts.append(time.perf_counter()-t)
    peak=torch.cuda.max_memory_allocated()/1024**2 if dev.type=='cuda' else psutil.Process().memory_info().rss/1024**2
    row={'total_parameters':total,'trainable_parameters':trainable,'mean_inference_ms':1000*np.mean(ts),'p95_inference_ms':1000*np.quantile(ts,.95),'throughput_images_s':1/np.mean(ts),'peak_memory_mb':peak,'device':str(dev)}
    pd.DataFrame([row]).to_csv('complexity_results.csv',index=False); return row
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); a=ap.parse_args(); print(measure(load_config(a.config)))
